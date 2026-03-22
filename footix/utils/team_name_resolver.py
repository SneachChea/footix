"""Robust team name resolver for matching calendar names to training names.

Combines a static YAML mapping (hand-curated, per-league) with rapidfuzz
fuzzy matching (WRatio). When the fuzzy confidence is insufficient and the
resolver is in interactive mode, the user is prompted to confirm or provide the
correct training name. Confirmed mappings are persisted back to the YAML so
subsequent runs require no interaction for the same team names.

Usage:
    >>> from footix.utils.team_name_resolver import TeamNameResolver
    >>> resolver = TeamNameResolver(league="ligue_1", interactive=True)
    >>> mapping = resolver.resolve(calendar_names=[...], training_names=[...])

"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

# Default mapping directory: <repo_root>/data/team_name_mappings/
_DEFAULT_MAPPING_DIR = Path(__file__).parent.parent.parent / "data" / "team_name_mappings"

# Map competition strings to league YAML keys
COMPETITION_TO_LEAGUE_KEY: dict[str, str] = {
    "FRA Ligue 1": "ligue_1",
    "FRA Ligue 2": "ligue_2",
    "DEU Bundesliga 1": "bundesliga_1",
}


class UnresolvedTeamNameError(ValueError):
    """Raised when a team name cannot be resolved in non-interactive mode.

    Attributes:
        team_name: The unresolved calendar team name.
        candidates: Top fuzzy candidates as (name, score) tuples.
    """

    def __init__(
        self,
        message: str,
        team_name: str,
        candidates: list[tuple[str, float]],
    ) -> None:
        """Initialize UnresolvedTeamNameError.

        Args:
            message: Human-readable error description.
            team_name: The calendar name that failed to resolve.
            candidates: Top rapidfuzz candidates.
        """
        super().__init__(message)
        self.team_name = team_name
        self.candidates = candidates


class TeamNameResolver:
    """Map calendar-format team names to model-training team names.

    Resolution priority:
        1. Static YAML mapping (case-insensitive exact match).
        2. Exact case-insensitive match in training names.
        3. rapidfuzz WRatio ≥ ``auto_threshold``  → auto-accept.
        4. rapidfuzz WRatio ≥ ``confirm_threshold`` → interactive confirm (if
           ``interactive=True``) or ``UnresolvedTeamNameError`` otherwise.
        5. Below ``confirm_threshold`` → interactive selection from top-5
           candidates or ``UnresolvedTeamNameError`` in non-interactive mode.

    New mappings discovered during a run are persisted to the YAML file so
    subsequent runs skip the interactive step for the same team names.

    Args:
        league: League YAML key, e.g. ``"ligue_1"``, ``"ligue_2"``,
            ``"bundesliga_1"``.  Can also be a full competition string such as
            ``"FRA Ligue 1"`` — it will be converted automatically.
        mapping_dir: Directory containing ``<league>.yaml`` mapping files.
            Defaults to ``<repo_root>/data/team_name_mappings/``.
        interactive: When ``True``, ambiguous matches trigger a CLI prompt.
            Set to ``False`` for CI / non-interactive pipelines;
            ``UnresolvedTeamNameError`` is raised instead.
        auto_threshold: rapidfuzz WRatio score (0–100) above which a match is
            accepted automatically.  Default ``90``.
        confirm_threshold: rapidfuzz WRatio score (0–100) above which a match
            is proposed for interactive confirmation.  Default ``70``.
    """

    def __init__(
        self,
        league: str,
        mapping_dir: Path | None = None,
        interactive: bool = True,
        auto_threshold: float = 90.0,
        confirm_threshold: float = 70.0,
    ) -> None:
        """Initialize TeamNameResolver.

        Args:
            league: League key or full competition string.
            mapping_dir: Optional override for the YAML directory.
            interactive: Enable interactive CLI prompts for ambiguous names.
            auto_threshold: WRatio score for automatic acceptance.
            confirm_threshold: WRatio score for interactive confirmation.
        """
        # Accept both "FRA Ligue 1" and "ligue_1" as league identifier
        self._league = COMPETITION_TO_LEAGUE_KEY.get(league, league)
        self._mapping_dir = mapping_dir or _DEFAULT_MAPPING_DIR
        self._interactive = interactive
        self._auto_threshold = auto_threshold
        self._confirm_threshold = confirm_threshold
        self._yaml_path = self._mapping_dir / f"{self._league}.yaml"
        self._static_map: dict[str, str] = self._load_static_map()
        self._new_mappings: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_static_map(self) -> dict[str, str]:
        """Load mappings from the YAML file.

        Returns:
            Dict mapping calendar names (stripped) to training names.
        """
        if not self._yaml_path.exists():
            return {}
        with open(self._yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_mappings: dict[str, str] = data.get("mappings") or {}
        return {k.strip(): v for k, v in raw_mappings.items()}

    def _check_static(self, name: str) -> str | None:
        """Look up *name* in the static YAML mapping (case-insensitive).

        Args:
            name: Calendar team name.

        Returns:
            Mapped training name, or ``None`` if not found.
        """
        # Exact match first (fastest)
        if name in self._static_map:
            return self._static_map[name]
        # Case-insensitive fallback
        lower = name.strip().lower()
        for key, value in self._static_map.items():
            if key.strip().lower() == lower:
                return value
        return None

    def _fuzzy_candidates(
        self, name: str, candidates: list[str], limit: int = 5
    ) -> list[tuple[str, float]]:
        """Compute rapidfuzz WRatio scores between *name* and *candidates*.

        WRatio is token-order invariant and handles abbreviations and partial
        matches, making it well suited for football club names.

        Args:
            name: Query team name.
            candidates: Pool of training team names.
            limit: Maximum number of results to return.

        Returns:
            List of (candidate_name, score) sorted by descending score.
        """
        results = process.extract(
            name,
            candidates,
            scorer=fuzz.WRatio,
            limit=limit,
            score_cutoff=0,
        )
        return [(match, float(score)) for match, score, _ in results]

    def _interactive_confirm(self, name: str, candidates: list[tuple[str, float]]) -> str:
        """Prompt the user to resolve an ambiguous team name.

        Displays top candidates with their scores and asks the user to pick one,
        type a custom name, or skip (which keeps the original calendar name).

        Args:
            name: Unresolved calendar team name.
            candidates: Top (candidate, score) tuples from rapidfuzz.

        Returns:
            Resolved training name chosen or typed by the user.
        """
        top = candidates[:5]
        print(f"\n{'─' * 60}", file=sys.stderr)
        print(f"⚠️  Cannot auto-resolve: '{name}'", file=sys.stderr)
        print("   Top candidates (rapidfuzz WRatio score / 100):", file=sys.stderr)
        for i, (candidate, score) in enumerate(top, 1):
            print(f"   [{i}] {candidate!r:<30} score: {score:.0f}", file=sys.stderr)
        print("   [0] Enter the correct name manually", file=sys.stderr)
        print(
            "   [s] Skip — keep original name (will likely cause a model error)", file=sys.stderr
        )

        while True:
            try:
                raw = input("   Your choice: ").strip()
            except EOFError:
                # Non-interactive environment despite interactive=True
                return name

            if raw.lower() == "s":
                print(f"   ⚠️  Skipping '{name}' — using original name.", file=sys.stderr)
                return name
            if raw == "0":
                manual = input("   Type the correct training name: ").strip()
                if manual:
                    print(f"   ✅ Mapped '{name}' → '{manual}'", file=sys.stderr)
                    return manual
                continue
            try:
                idx = int(raw)
                if 1 <= idx <= len(top):
                    picked = top[idx - 1][0]
                    print(f"   ✅ Mapped '{name}' → '{picked}'", file=sys.stderr)
                    return picked
            except ValueError:
                pass
            print("   Please enter a number between 0 and 5, or 's' to skip.", file=sys.stderr)

    def _persist(self) -> None:
        """Append newly discovered mappings to the YAML file.

        Existing mappings are preserved; only new entries are added.
        Note: YAML comments are removed on rewrite.
        """
        if not self._new_mappings:
            return
        self._yaml_path.parent.mkdir(parents=True, exist_ok=True)

        if self._yaml_path.exists():
            with open(self._yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        existing: dict[str, str] = data.get("mappings") or {}
        existing.update(self._new_mappings)
        data["mappings"] = dict(sorted(existing.items()))

        with open(self._yaml_path, "w", encoding="utf-8") as f:
            f.write(
                "# Team name mappings — auto-updated by TeamNameResolver\n"
                '# Format: "calendar_name": "training_name"\n\n'
            )
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        print(
            f"   💾 Persisted {len(self._new_mappings)} new mapping(s) → {self._yaml_path}",
            file=sys.stderr,
        )
        self._new_mappings = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        calendar_names: list[str],
        training_names: list[str],
    ) -> dict[str, str]:
        """Build a full mapping from *calendar_names* to *training_names*.

        Each calendar name goes through the resolution priority chain (static →
        exact → fuzzy auto → interactive / error) and the result is returned as
        a dictionary.  Any newly confirmed mappings are persisted to the YAML
        before this method returns.

        Args:
            calendar_names: Team names as they appear in the fixture/calendar CSV.
            training_names: Team names as they appear in the training dataset
                (football-data.co.uk convention).

        Returns:
            Dict mapping every calendar name to its resolved training name.
            Skipped names are mapped to themselves.

        Raises:
            UnresolvedTeamNameError: If ``interactive=False`` and a name cannot
                be auto-resolved above ``auto_threshold``.
        """
        mapping: dict[str, str] = {}

        for name in calendar_names:
            # 1. Static YAML lookup (exact or case-insensitive)
            static = self._check_static(name)
            if static is not None:
                mapping[name] = static
                continue

            # 2. Exact case-insensitive match in training names
            lower_name = name.strip().lower()
            exact = next(
                (t for t in training_names if t.strip().lower() == lower_name),
                None,
            )
            if exact is not None:
                mapping[name] = exact
                continue

            # 3. rapidfuzz fuzzy matching
            candidates = self._fuzzy_candidates(name, training_names)
            if not candidates:
                # Empty training set — keep original name
                mapping[name] = name
                continue

            best_match, best_score = candidates[0]

            # 3a. Auto-accept (high confidence)
            if best_score >= self._auto_threshold:
                print(
                    f"   🤖 Auto-matched: '{name}' → '{best_match}'  (score {best_score:.0f})",
                    file=sys.stderr,
                )
                mapping[name] = best_match
                self._new_mappings[name] = best_match
                continue

            # 3b. Confirm threshold — propose but ask for confirmation
            if best_score >= self._confirm_threshold:
                if self._interactive:
                    print(
                        f"\n{'─' * 60}\n"
                        f"🔍 Proposed match:\n"
                        f"   '{name}' → '{best_match}'\n"
                        f"   (score {best_score:.0f}/100)",
                        file=sys.stderr,
                    )
                    try:
                        ans = input("   Accept? [Y/n]: ").strip().lower()
                    except EOFError:
                        ans = ""
                    if ans in ("", "y", "yes"):
                        print(f"   ✅ Accepted '{name}' → '{best_match}'", file=sys.stderr)
                        mapping[name] = best_match
                        self._new_mappings[name] = best_match
                        continue
                    # user rejected — fall through to full interactive selection
                    resolved = self._interactive_confirm(name, candidates)
                    mapping[name] = resolved
                    self._new_mappings[name] = resolved
                    continue
                else:
                    raise UnresolvedTeamNameError(
                        f"Team '{name}' matched '{best_match}' with score {best_score:.0f} "
                        f"(below auto_threshold={self._auto_threshold}). "
                        f"Add the mapping to '{self._yaml_path}' or run interactively.",
                        name,
                        candidates,
                    )

            # 3c. Below confirm threshold — full manual selection
            if self._interactive:
                resolved = self._interactive_confirm(name, candidates)
                mapping[name] = resolved
                self._new_mappings[name] = resolved
            else:
                raise UnresolvedTeamNameError(
                    f"Team '{name}' could not be resolved (best candidate: "
                    f"'{best_match}' at {best_score:.0f}/100, threshold: "
                    f"{self._confirm_threshold}). "
                    f"Add the mapping to '{self._yaml_path}' or run interactively.",
                    name,
                    candidates,
                )

        self._persist()
        return mapping
