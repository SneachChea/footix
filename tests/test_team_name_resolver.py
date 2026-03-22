"""Tests for footix.utils.team_name_resolver.TeamNameResolver."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from footix.utils.team_name_resolver import TeamNameResolver, UnresolvedTeamNameError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LIGUE_1_TRAINING_TEAMS = [
    "Angers",
    "Auxerre",
    "Brest",
    "Le Havre",
    "Lens",
    "Lille",
    "Lorient",
    "Lyon",
    "Marseille",
    "Metz",
    "Monaco",
    "Nantes",
    "Nice",
    "Paris FC",
    "Paris SG",
    "Rennes",
    "Strasbourg",
    "Toulouse",
]

LIGUE_1_CALENDAR_TEAMS = [
    "AJ Auxerre",
    "AS Monaco",
    "Angers SCO",
    "FC Lorient",
    "FC Metz",
    "FC Nantes",
    "Havre Athletic Club",
    "LOSC Lille",
    "OGC Nice",
    "Olympique Lyonnais",
    "Olympique de Marseille",
    "Paris FC",
    "Paris Saint-Germain",
    "RC Lens",
    "RC Strasbourg Alsace",
    "Stade Brestois 29",
    "Stade Rennais FC",
    "Toulouse FC",
]

EXPECTED_L1_MAPPING = {
    "AJ Auxerre": "Auxerre",
    "AS Monaco": "Monaco",
    "Angers SCO": "Angers",
    "FC Lorient": "Lorient",
    "FC Metz": "Metz",
    "FC Nantes": "Nantes",
    "Havre Athletic Club": "Le Havre",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "Paris FC": "Paris FC",
    "Paris Saint-Germain": "Paris SG",
    "RC Lens": "Lens",
    "RC Strasbourg Alsace": "Strasbourg",
    "Stade Brestois 29": "Brest",
    "Stade Rennais FC": "Rennes",
    "Toulouse FC": "Toulouse",
}

LIGUE_2_TRAINING_TEAMS = [
    "Amiens",
    "Annecy",
    "Bastia",
    "Boulogne",
    "Clermont",
    "Dunkerque",
    "Grenoble",
    "Guingamp",
    "Laval",
    "Le Mans",
    "Montpellier",
    "Nancy",
    "Pau FC",
    "Red Star",
    "Reims",
    "Rodez",
    "St Etienne",
    "Troyes",
]


# ---------------------------------------------------------------------------
# Tests — YAML static mapping (Ligue 1 known names)
# ---------------------------------------------------------------------------


class TestStaticMapLigue1:
    """All Ligue 1 calendar names must resolve via the static YAML."""

    def test_all_ligue1_calendar_names_resolved(self) -> None:
        """Resolve all 18 Ligue 1 calendar teams against training names."""
        resolver = TeamNameResolver(league="ligue_1", interactive=False)
        mapping = resolver.resolve(
            calendar_names=LIGUE_1_CALENDAR_TEAMS,
            training_names=LIGUE_1_TRAINING_TEAMS,
        )
        for calendar_name, expected_training_name in EXPECTED_L1_MAPPING.items():
            assert mapping[calendar_name] == expected_training_name, (
                f"'{calendar_name}' → expected '{expected_training_name}', "
                f"got '{mapping[calendar_name]}'"
            )

    def test_exact_match_no_change(self) -> None:
        """Names that are identical in both sources are returned unchanged."""
        resolver = TeamNameResolver(league="ligue_1", interactive=False)
        mapping = resolver.resolve(
            calendar_names=["Paris FC"],
            training_names=LIGUE_1_TRAINING_TEAMS,
        )
        assert mapping["Paris FC"] == "Paris FC"

    def test_case_insensitive_static_lookup(self) -> None:
        """Static YAML lookup is case-insensitive."""
        resolver = TeamNameResolver(league="ligue_2", interactive=False)
        # "pau" is in ligue_2.yaml mapped to "Pau FC"
        mapping = resolver.resolve(
            calendar_names=["pau"],
            training_names=LIGUE_2_TRAINING_TEAMS,
        )
        assert mapping["pau"] == "Pau FC"


# ---------------------------------------------------------------------------
# Tests — rapidfuzz auto-accept (names NOT in static YAML, high similarity)
# ---------------------------------------------------------------------------


class TestFuzzyAutoAccept:
    """Names with WRatio ≥ auto_threshold must be resolved automatically."""

    def test_abbreviated_name_auto_matched(self, tmp_path: Path) -> None:
        """'Stade Rennais' (no 'FC') should auto-match 'Rennes'."""
        mapping_dir = tmp_path / "mappings"
        mapping_dir.mkdir()
        resolver = TeamNameResolver(
            league="ligue_1",
            mapping_dir=mapping_dir,
            interactive=False,
            auto_threshold=65,  # WRatio('Stade Rennais', 'Rennes') ≈ 68
            confirm_threshold=40,
        )
        training = ["Rennes", "Lyon", "Monaco"]
        mapping = resolver.resolve(
            calendar_names=["Stade Rennais"],
            training_names=training,
        )
        assert mapping["Stade Rennais"] == "Rennes"

    def test_transposed_name_auto_matched(self, tmp_path: Path) -> None:
        """'Bayern Munchen' should auto-match 'Bayern Munich' at high threshold."""
        mapping_dir = tmp_path / "mappings"
        mapping_dir.mkdir()
        resolver = TeamNameResolver(
            league="bundesliga_1",
            mapping_dir=mapping_dir,
            interactive=False,
            auto_threshold=85,
        )
        training = ["Bayern Munich", "Dortmund", "Leverkusen"]
        mapping = resolver.resolve(
            calendar_names=["Bayern Munchen"],
            training_names=training,
        )
        assert mapping["Bayern Munchen"] == "Bayern Munich"


# ---------------------------------------------------------------------------
# Tests — non-interactive error handling
# ---------------------------------------------------------------------------


class TestNonInteractiveErrors:
    """Unresolvable names in non-interactive mode must raise UnresolvedTeamNameError."""

    def test_completely_unknown_name_raises(self) -> None:
        """A completely unknown name raises UnresolvedTeamNameError."""
        resolver = TeamNameResolver(
            league="ligue_1",
            interactive=False,
            auto_threshold=99,  # force failure for any fuzzy match
            confirm_threshold=98,
        )
        with pytest.raises(UnresolvedTeamNameError) as exc_info:
            resolver.resolve(
                calendar_names=["Equipe Inconnue XXXX"],
                training_names=["Lyon", "Rennes", "Monaco"],
            )
        assert exc_info.value.team_name == "Equipe Inconnue XXXX"
        assert len(exc_info.value.candidates) > 0

    def test_error_contains_candidates(self) -> None:
        """UnresolvedTeamNameError must carry top candidates."""
        resolver = TeamNameResolver(
            league="ligue_1",
            interactive=False,
            auto_threshold=99,
            confirm_threshold=98,
        )
        with pytest.raises(UnresolvedTeamNameError) as exc_info:
            resolver.resolve(
                calendar_names=["Fake Team Alpha"],
                training_names=["Lyon", "Rennes", "Monaco", "Nice", "Lens"],
            )
        candidates = exc_info.value.candidates
        assert isinstance(candidates, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)


# ---------------------------------------------------------------------------
# Tests — YAML persistence
# ---------------------------------------------------------------------------


class TestYamlPersistence:
    """New mappings must be written back to the YAML file."""

    def test_new_auto_mapping_persisted(self, tmp_path: Path) -> None:
        """A newly auto-resolved mapping is persisted to the YAML."""
        mapping_dir = tmp_path / "mappings"
        mapping_dir.mkdir()
        yaml_file = mapping_dir / "test_league.yaml"

        # Resolver with low threshold so the fuzzy match auto-accepts
        resolver = TeamNameResolver(
            league="test_league",
            mapping_dir=mapping_dir,
            interactive=False,
            auto_threshold=60,
            confirm_threshold=40,
        )
        training = ["Bayern Munich", "Dortmund"]
        resolver.resolve(
            calendar_names=["Bayern Munchen"],
            training_names=training,
        )

        # YAML must have been created / updated
        assert yaml_file.exists(), "YAML file was not created by _persist()"
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "Bayern Munchen" in data.get("mappings", {})
        assert data["mappings"]["Bayern Munchen"] == "Bayern Munich"

    def test_existing_static_not_overwritten(self, tmp_path: Path) -> None:
        """Pre-existing static mappings survive a persist() call."""
        mapping_dir = tmp_path / "mappings"
        mapping_dir.mkdir()
        yaml_file = mapping_dir / "test_league.yaml"
        # Pre-populate
        yaml_file.write_text(
            "mappings:\n  'Existing Team': 'Correct Name'\n",
            encoding="utf-8",
        )

        resolver = TeamNameResolver(
            league="test_league",
            mapping_dir=mapping_dir,
            interactive=False,
            auto_threshold=55,
            confirm_threshold=40,
        )
        training = ["Correct Name", "Other Club"]
        # Trigger a new auto-resolution to force persist
        resolver.resolve(
            calendar_names=["Existing Team", "Other Clubb"],
            training_names=training,
        )

        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["mappings"].get("Existing Team") == "Correct Name"


# ---------------------------------------------------------------------------
# Tests — competition string as league identifier
# ---------------------------------------------------------------------------


class TestCompetitionStringInput:
    """Full competition strings such as 'FRA Ligue 1' must work as league IDs."""

    def test_competition_string_resolves_ligue_1(self) -> None:
        """'FRA Ligue 1' is converted to 'ligue_1' YAML key transparently."""
        resolver = TeamNameResolver(league="FRA Ligue 1", interactive=False)
        mapping = resolver.resolve(
            calendar_names=["Olympique Lyonnais"],
            training_names=LIGUE_1_TRAINING_TEAMS,
        )
        assert mapping["Olympique Lyonnais"] == "Lyon"

    def test_competition_string_resolves_ligue_2(self) -> None:
        """'FRA Ligue 2' is converted to 'ligue_2' YAML key transparently."""
        resolver = TeamNameResolver(league="FRA Ligue 2", interactive=False)
        mapping = resolver.resolve(
            calendar_names=["Estac Troyes"],
            training_names=LIGUE_2_TRAINING_TEAMS,
        )
        assert mapping["Estac Troyes"] == "Troyes"
