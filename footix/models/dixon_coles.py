import logging

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import footix.models.score_matrix as score_matrix
import footix.models.utils as model_utils
from footix.utils.decorators import verify_required_column

logger = logging.getLogger(name=__name__)


# TODO update the loss function to include the correlation network
class NeuralDixonColes:
    def __init__(self, n_teams: int, n_goals: int) -> None:
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.gamma = nn.Parameter(torch.tensor(0.3))  # Home advantage
        self.alphas = nn.Parameter(torch.randn(n_teams))  # Attack strength
        self.betas = nn.Parameter(torch.randn(n_teams))  # Defense strength
        self.correlation = CorrectionNetwork(hidden_dim=16)  # Correlation network

    @verify_required_column(column_names={"home_team", "away_team", "ftr", "fthg", "ftag"})
    def fit(self, X_train: pd.DataFrame) -> None:
        self.dict_teams = self.mapping_team_index(X_train["home_team"])
        self._sanity_check(X_train["away_team"])
        goals_home, mask_home = model_utils.compute_goals_home_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        goals_away, mask_away = model_utils.compute_goals_away_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        # convertion to tensor
        goals_home, mask_home, goals_away, mask_away = model_utils.to_torch_tensor(
            goals_home, mask_home, goals_away, mask_away
        )
        optimizer = torch.optim.AdamW(
            [self.gamma, self.alphas, self.betas] + list(self.correlation.parameters()), lr=0.01
        )

        with tqdm(total=3_000) as pbar:
            for epoch in range(3_000):
                optimizer.zero_grad()
                loss = self.dixon_coles_likelihood(goals_home, goals_away, mask_home, mask_away)
                penalty = self.alphas - torch.mean(self.alphas)
                penalty += torch.mean(self.betas) - self.betas
                loss += torch.sum(penalty)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.5f}")
                pbar.update(1)

    def print_parameters(self) -> None:
        str_gamma = f"Gamma = {self.gamma}\n"
        str_alpha = "".join(
            [f"alpha team-{team} = {self.alphas[idx]}\n" for team, idx in self.dict_teams.items()]
        )
        str_beta = "".join(
            [f"beta team-{team} = {self.betas[idx]}\n" for team, idx in self.dict_teams.items()]
        )
        print(str_gamma + str_alpha + str_beta)

    def predict(self, home_team: str, away_team: str) -> score_matrix.GoalMatrix:
        if home_team not in self.dict_teams.keys():
            raise ValueError(f"Home team {home_team} is not in the list.")
        if away_team not in self.dict_teams.keys():
            raise ValueError(f"Away team {away_team} is not in the list.")
        i = self.dict_teams[home_team]
        j = self.dict_teams[away_team]
        lamb = torch.exp(self.alphas[i] + self.betas[j] + self.gamma).cpu().detach().item()
        mu = torch.exp(self.alphas[j] + self.betas[i]).cpu().detach().item()
        rho_correction = self.compute_correlation_matrix().detach().cpu().numpy()
        return score_matrix.GoalMatrix(
            home_probs=model_utils.poisson_proba(lambda_param=lamb, k=self.n_goals),
            away_probs=model_utils.poisson_proba(lambda_param=mu, k=self.n_goals),
            correlation_matrix=rho_correction,
        )

    def mapping_team_index(self, teams: pd.Series) -> dict[str, int]:
        list_teams = list(sorted(teams.unique()))
        return {element: index for index, element in enumerate(list_teams)}

    def _sanity_check(self, teams: pd.Series) -> None:
        dict_teams_away = self.mapping_team_index(teams)
        if self.dict_teams != dict_teams_away:
            raise ValueError(
                "Not every teams have played at home and away. Please give another dataset."
            )
        if len(self.dict_teams) != self.n_teams:
            raise ValueError(f"Expecting {self.n_teams} teams, only got {len(self.dict_teams)}.")

    def dixon_coles_likelihood(
        self,
        goals_home: torch.Tensor,
        goals_away: torch.Tensor,
        basis_home: torch.Tensor,
        basis_away: torch.Tensor,
    ) -> torch.Tensor:
        log_lamdas = (
            torch.matmul(basis_home, self.alphas)
            + torch.matmul(basis_away, self.betas)
            + self.gamma
        )
        log_mus = torch.matmul(basis_away, self.alphas) + torch.matmul(basis_home, self.betas)
        lambdas = torch.exp(log_lamdas)
        mus = torch.exp(log_mus)
        corr = self.correlation(torch.stack([goals_home, goals_away], dim=1))
        log = lambdas + mus - goals_home * log_lamdas - goals_away * log_mus - corr
        return torch.mean(log)

    def compute_correlation_matrix(self) -> torch.Tensor:
        n = self.n_goals
        matrix = torch.zeros((n, n))
        with torch.inference_mode():
            for i in range(n):
                for j in range(n):
                    goals = torch.tensor([[float(i), float(j)]])
                    matrix[i, j] = torch.exp(self.correlation.forward(goals))
        return matrix


# 1. Réseau de correction f(k,l; theta)
class CorrectionNetwork(nn.Module):
    def __init__(self, hidden_dim=8):
        super(CorrectionNetwork, self).__init__()
        # Un réseau simple à 2 couches
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, k_l):
        # k_l : tenseur de forme (N, 2) contenant [k, l] en float.
        return torch.tanh(self.net(k_l).squeeze(1))  # renvoie un tenseur de forme (N,)
