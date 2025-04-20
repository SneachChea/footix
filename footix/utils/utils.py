from typing import Final, TypeVar

import numpy as np

import footix.models.basic_poisson as model_poisson
import footix.models.dixon_coles as dixon_coles

T = TypeVar("T", model_poisson.BasicPoisson, dixon_coles.NeuralDixonColes)

EPS: Final[float] = 1e-5

DICO_COMPATIBILITY: Final[dict[str, str]] = {
    "AC Ajaccio": "Ajaccio",
    "Lyon": "Lyon",
    "Rennes": "Rennes",
    "Paris SG": "Paris SG",
    "Nantes": "Nantes",
    "Lens": "Lens",
    "Troyes": "Troyes",
    "Reims": "Reims",
    "Nice": "Nice",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Strasbourg": "Strasbourg",
    "Lille": "Lille",
    "Auxerre": "Auxerre",
    "Brest": "Brest",
    "Clermont": "Clermont",
    "Lorient": "Lorient",
    "Angers": "Angers",
    "Monaco": "Monaco",
    "Marseille": "Marseille",
    "Sochaux": "Sochaux",
    "Bastia": "Bastia",
    "Valenciennes": "Valenciennes",
    "Guingamp": "Guingamp",
    "Nîmes": "Nimes",
    "St Etienne": "St Etienne",
    "Grenoble": "Grenoble",
    "Rodez": "Rodez",
    "Metz": "Metz",
    "Pau": "Pau FC",
    "Laval": "Laval",
    "Grenoble": "Grenoble",
    "Bordeaux": "Bordeaux",
    "Caen": "Caen",
    "Paris FC": "Paris FC",
    "Dijon": "Dijon",
    "Quevilly Rouen": "Quevilly Rouen",
    "Niort": "Niort",
    "Amiens": "Amiens",
    "Le Havre": "Le Havre",
    "Annecy FC": "Annecy",
}


def poisson_model_recap(home_team: str, away_team: str, model: T) -> None:
    score_recap = model.predict(home_team=home_team, away_team=away_team)
    proba_h, proba_d, proba_a = score_recap.return_probas()
    alpha = model.alphas
    beta = model.betas
    index_home = model.dict_teams[home_team]
    index_away = model.dict_teams[away_team]
    alpha_sorted = sorted(enumerate(alpha), key=lambda x: x[1], reverse=True)
    beta_sorted = sorted(enumerate(beta), key=lambda x: x[1], reverse=False)
    power_attack_home = [index for index, _ in alpha_sorted].index(index_home)
    power_attack_away = [index for index, _ in alpha_sorted].index(index_away)
    power_defence_home = [index for index, _ in beta_sorted].index(index_home)
    power_defence_away = [index for index, _ in beta_sorted].index(index_away)

    max_index = np.unravel_index(score_recap.matrix_array.argmax(), score_recap.matrix_array.shape)

    print(f"{home_team} vs {away_team} \n")
    print(
        f"team {home_team} has the {1+power_attack_home}-th attack power over {model.n_teams}"
        " teams"
    )
    print(
        f"team {away_team} has the {1+power_attack_away}-th attack power over {model.n_teams}"
        " teams."
    )
    print(
        f"team {home_team} has the {1+power_defence_home}-th defense power over {model.n_teams}"
        " teams."
    )
    print(
        f"team {away_team} has the {1+power_defence_away}-th defense power over {model.n_teams}"
        " teams"
    )

    print("#" * 4, "PROBABILITIES", "#" * 4)
    print(f"Probability of {home_team} win : {round(100*proba_h)} %")
    print(f"Probability of a draw : {round(100*proba_d)} %")
    print(f"Probability of {away_team} win : {round(100*proba_a)} %")
    print(f"Probability of less than 1.5 goals : {round(100*score_recap.less_15_goals())} %")
    print(f"Probability of more than 1.5 goals : {round(100*score_recap.more_15_goals())} %")
    print(f"Probability of less than 2.5 goals : {round(100*score_recap.less_25_goals())} %")
    print(f"Probability of more than 2.5 goals : {round(100*score_recap.more_25_goals())} %")
    print(f"Most likely score {max_index[0]}-{max_index[1]}")
    score_recap.visualize()
