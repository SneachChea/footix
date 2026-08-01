Data sources and data contracts
===============================

footix loads two kinds of data:

* **Historical match results**, used to train models.
* **Upcoming fixtures**, used to generate predictions before a matchday.

Every provider is a thin wrapper around a public source. This page describes
the competition keys, the season formats, the caching behavior, and the
columns each model expects.

Providers
---------

.. list-table::
   :header-rows: 1

   * - Provider
     - Class
     - Data
     - Caching
   * - football-data.co.uk
     - :class:`~footix.data_io.footballdata.ScrapFootballData`
     - Historical results (full season)
     - CSV on disk; ``force_reload=True`` re-downloads
   * - understat.com
     - :class:`~footix.data_io.understat.ScrapUnderstat`
     - Historical results with xG and forecasts
     - In-process ``lru_cache`` (``force_reload`` currently ignored)
   * - football-data.org
     - :class:`~footix.data_io.football_data_org.ScrapFootballDataOrg`
     - Upcoming fixtures (next matchday)
     - Disk cache with TTL (6 h by default); ``force_reload`` bypasses it
   * - APIFootball.com
     - :class:`~footix.data_io.apifootball_com.ScrapAPIFootballCom`
     - Upcoming fixtures (next matchday), Ligue 2 only
     - Disk cache with TTL (6 h by default); ``force_reload`` bypasses it

Competition keys
----------------

Providers are configured with the *footix competition keys* defined in
``footix.data_io.utils_scrapper.MAPPING_COMPETITIONS`` — not with raw
provider codes. For example ``"FRA Ligue 1"`` maps to the football-data.co.uk
slug ``F1``, the understat slug ``Ligue_1`` and the football-data.org code
``FL1``.

.. list-table::
   :header-rows: 1

   * - Key
     - football-data.co.uk
     - understat
     - football-data.org
     - APIFootball.com
   * - ``FRA Ligue 1``
     - ``F1``
     - ``Ligue_1``
     - ``FL1``
     - —
   * - ``FRA Ligue 2``
     - ``F2``
     - —
     - —
     - ``164``
   * - ``ENG Premier League``
     - ``E0``
     - ``EPL``
     - ``PL``
     - —
   * - ``ENG Championship``
     - ``E1``
     - —
     - ``ELC``
     - —
   * - ``DEU Bundesliga 1``
     - ``D1``
     - ``Bundesliga``
     - ``BL1``
     - —
   * - ``DEU Bundesliga 2``
     - ``D2``
     - —
     - —
     - —
   * - ``ITA Serie A``
     - ``I1``
     - ``Serie_A``
     - ``SA``
     - —
   * - ``ITA Serie B``
     - ``I2``
     - —
     - —
     - —
   * - ``SPA La Liga``
     - ``SP1``
     - ``La_Liga``
     - ``PD``
     - —
   * - ``SPA La Liga 2``
     - ``SP2``
     - —
     - —
     - —

Season formats
--------------

* **football-data.co.uk / understat** accept ``"2024-2025"``, ``"2024/2025"``
  or the compact ``"2425"``. The value is normalized to the provider URL format
  (``2425`` for football-data.co.uk).
* **football-data.org** expects the season as a 4-digit starting year
  (``"2026"`` for the 2026-2027 season). APIFootball.com does not take a
  season argument: it fetches whatever is scheduled in the next 15 days.

Output columns
--------------

Historical scrapers return snake_case columns, including at least
``date``, ``home_team``, ``away_team``, ``fthg``, ``ftag``, ``ftr`` and a
stable ``match_id`` built by :func:`~footix.data_io.utils_scrapper.add_match_id`
in the form ``"Home - Away - YYYY-MM-DD"``.

Upcoming-fixture scrapers return the same core columns plus
``kickoff`` (Europe/Paris timezone), ``status``, ``gameweek`` and
``source_fixture_id``. Team names come from the provider calendar and must
be mapped to your training names before prediction.

Required columns per model
--------------------------

* :class:`~footix.models.basic_poisson.PoissonModel`:
  ``home_team``, ``away_team``, ``fthg``, ``ftag``, ``ftr``
  (``ftr`` is required by validation but not used in the likelihood).
* :class:`~footix.models.elo.EloDavidson` (via
  :class:`~footix.data_io.data_reader.EloDataReader`):
  ``date``, ``home_team``, ``away_team``, ``fthg``, ``ftag``, ``ftr``.
  Dates are parsed with ``dayfirst=True`` and the data is sorted by date.
* :class:`~footix.models.bayesian.BayesianModel`:
  ``home_team``, ``away_team``, ``fthg``, ``ftag``.

Missing columns raise a clear error via the
:func:`~footix.utils.decorators.verify_required_column` decorator.

Team name normalization
-----------------------

Calendar names ("Paris Saint-Germain") rarely match training names
("PSG"). Pass a ``mapping_teams`` dict to any scraper, or use
:class:`~footix.utils.team_name_resolver.TeamNameResolver`, which combines a
static YAML map (``data/team_name_mappings/``), fuzzy matching and
persistence of new mappings.

Credentials
-----------

football-data.org and APIFootball.com require a token/key. Pass them as
constructor arguments, typically read from the environment — never hard-code
them:

.. code-block:: python

   import os

   from footix.data_io.apifootball_com import ScrapAPIFootballCom
   from footix.data_io.football_data_org import ScrapFootballDataOrg

   ligue1_fixtures = ScrapFootballDataOrg(
       competition="FRA Ligue 1",
       season="2026",
       api_token=os.environ["FOOTBALL_DATA_ORG_TOKEN"],
       path="./data",
   ).get_fixtures()

   ligue2_fixtures = ScrapAPIFootballCom(
       competition="FRA Ligue 2",
       api_key=os.environ["APIFOOTBALL_COM_KEY"],
       path="./data",
   ).get_fixtures()

Offline workflow
----------------

For reproducible experiments, download a season once (``force_reload=False``)
and reuse the cached CSV. The tutorials do exactly this — see
:doc:`../tutorials/elo` and :doc:`../tutorials/poisson`.
