# Architecture Refactor Plan — Modern Data Stack

## Goal
Transform the Football Bets project from a single-file Streamlit + CSV app into a
production-grade, portfolio-ready data engineering project using:
**MinIO → Dremio → dbt → PostgreSQL → Streamlit**

---

## New Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Data Sources                          │
│  football-data.co.uk  │  ESPN  │  API-Football  │  ...  │
└──────────────────┬───────────────────────────────────────┘
                   │  ingestion layer (Python)
                   ▼
┌──────────────────────────────────────────────────────────┐
│               MinIO  (Object Storage / Data Lake)         │
│                                                           │
│  fb-raw/          fb-processed/        fb-models/         │
│  football-data/   Parquet (bronze)     xgboost_*.pkl      │
│  espn/            dbt output (silver)  model_meta.json    │
└──────────────────┬───────────────────────────────────────┘
                   │  S3-compatible connector
                   ▼
┌──────────────────────────────────────────────────────────┐
│               Dremio  (Federated Query Engine)            │
│                                                           │
│  Source 1: MinIO  (fb-raw, fb-processed, fb-dbt)         │
│  Source 2: PostgreSQL  (config, metadata, injuries)       │
│                                                           │
│  Unified SQL interface over both sources                  │
└──────────────────┬───────────────────────────────────────┘
                   │  dbt-dremio adapter
                   ▼
┌──────────────────────────────────────────────────────────┐
│               dbt  (Transformation Layer)                 │
│                                                           │
│  staging/     → stg_matches (clean types, rename cols)   │
│  intermediate/→ int_team_form, int_h2h, int_features     │
│  marts/       → mart_standings, mart_context, mart_preds │
│                                                           │
│  Output: Parquet in MinIO fb-dbt/ via Dremio             │
└──────────────────┬───────────────────────────────────────┘
                   │
       ┌───────────┴──────────────┐
       ▼                          ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│   PostgreSQL      │   │     Streamlit  (UI Layer)         │
│                   │   │                                   │
│  app_config       │   │  Query Dremio SQL (mart tables)   │
│  refresh_metadata │   │  Read config from PostgreSQL      │
│  injury_reports   │   │  Load XGBoost models from MinIO   │
│  player_contribs  │   │  st.cache_resource for connections│
│  dbt_run_log      │   │                                   │
└──────────────────┘   └──────────────────────────────────┘
```

---

## Docker Compose Services

```yaml
services:
  postgres:
    image: postgres:16
    volumes: [postgres-data:/var/lib/postgresql/data]
    healthcheck: pg_isready

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    volumes: [minio-data:/data]
    ports: [9000, 9001 (console)]

  minio-setup:
    image: minio/mc
    depends_on: [minio]
    # one-shot: creates fb-raw, fb-processed, fb-models buckets

  dremio:
    image: dremio/dremio-oss:latest
    depends_on: [minio, postgres]
    ports: [9047 (UI/REST), 32010 (Arrow Flight)]
    volumes: [dremio-data:/opt/dremio/data]
    healthcheck: curl http://localhost:9047/apiv2/server_status

  dbt-runner:
    build: ./dbt
    depends_on:
      dremio: { condition: service_healthy }
    # Runs: dbt deps && dbt seed && dbt run && dbt test
    # Exits 0 on success — re-run manually or via cron

  model-trainer:
    build: ./sports_betting
    depends_on:
      dremio: { condition: service_healthy }
      minio:  { condition: service_healthy }
    # Runs train_models.py — trains XGBoost, pickles to MinIO fb-models/
    # Runs after dbt-runner completes

  streamlit:
    build: .
    depends_on: [dremio, postgres, minio]
    ports: [8501]
    env_file: .env

volumes:
  postgres-data:
  minio-data:
  dremio-data:
```

Single `docker-compose.yml` at repo root → `docker compose up` spins everything.

---

## MinIO Buckets

| Bucket           | Contents                                      | Layer       |
|------------------|-----------------------------------------------|-------------|
| `fb-raw`         | CSV downloads from football-data.co.uk        | Ingest      |
| `fb-bronze`      | Cleaned Parquet — output of fetch pipeline    | Bronze      |
| `fb-silver`      | dbt staging + intermediate Parquet tables     | Silver      |
| `fb-gold`        | dbt mart Parquet tables — queried by Streamlit| Gold        |
| `fb-models`      | XGBoost `.pkl` artifacts + `metadata.json`    | ML          |

**dbt Materialization Strategy:**
dbt-dremio supports two output modes:
- **Dremio Spaces** (virtual tables, in-memory) — fast for development, not persisted
- **Iceberg tables backed by MinIO** — persistent, queryable after dbt run, preferred for production

We use **Iceberg on MinIO** for all mart models so the data survives container restarts.
Staging and intermediate models use `view` materialization (no storage cost).

---

## dbt Project Structure

```
dbt/
├── dbt_project.yml
├── profiles.yml            # Dremio connection via Arrow Flight
├── packages.yml            # dbt-dremio adapter
├── models/
│   ├── staging/
│   │   └── stg_matches.sql         # raw CSV → clean types, snake_case cols
│   ├── intermediate/
│   │   ├── int_team_form.sql       # last N matches per team (replaces team_last5_form())
│   │   ├── int_h2h.sql             # head-to-head history (replaces h2h_features_for_match())
│   │   └── int_match_features.sql  # gap features (replaces build_match_training_data())
│   └── marts/
│       ├── mart_league_standings.sql   # replaces build_team_snapshot()
│       ├── mart_match_context.sql      # replaces build_context()
│       └── mart_player_contributions.sql
├── seeds/
│   └── team_name_map.csv           # replaces team_names.py TEAM_NAME_MAP dict
└── tests/
    ├── stg_matches__result_values.sql
    └── mart_standings__no_nulls.sql
```

---

## PostgreSQL Schema

```sql
-- App configuration (league list, momentum window, etc.)
CREATE TABLE app_config (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Replaces refresh_metadata.json
CREATE TABLE refresh_metadata (
    source      TEXT PRIMARY KEY,
    last_fetch  TIMESTAMPTZ,
    records     INT,
    status      TEXT,     -- 'ok' | 'error'
    error       TEXT
);

-- Replaces data/sports/external/injuries.csv
CREATE TABLE injury_reports (
    id               SERIAL PRIMARY KEY,
    team             TEXT NOT NULL,
    player           TEXT NOT NULL,
    injury_date      DATE,
    expected_return  DATE,
    importance_score FLOAT DEFAULT 1.0,
    source           TEXT,
    created_at       TIMESTAMPTZ DEFAULT now()
);

-- Replaces data/sports/external/player_contributions.csv
CREATE TABLE player_contributions (
    id               SERIAL PRIMARY KEY,
    team             TEXT NOT NULL,
    player           TEXT NOT NULL,
    season           TEXT,
    goals            INT  DEFAULT 0,
    assists          INT  DEFAULT 0,
    xG               FLOAT DEFAULT 0,
    importance_score FLOAT DEFAULT 1.0,
    source           TEXT,
    updated_at       TIMESTAMPTZ DEFAULT now()
);

-- dbt run audit log
CREATE TABLE dbt_run_log (
    id               SERIAL PRIMARY KEY,
    model_name       TEXT,
    run_at           TIMESTAMPTZ DEFAULT now(),
    status           TEXT,
    duration_seconds FLOAT,
    rows_affected    INT
);
```

---

## New Python Modules

| File                          | Responsibility                                    |
|-------------------------------|---------------------------------------------------|
| `sports_betting/storage.py`   | MinIO client — upload/download Parquet + pickles  |
| `sports_betting/db.py`        | PostgreSQL client — config, metadata, injuries    |
| `sports_betting/dremio.py`    | Dremio query client (Arrow Flight / REST)         |

### Environment Variables (.env at repo root)

```
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=football_bets
POSTGRES_USER=fbuser
POSTGRES_PASSWORD=fbpass

DREMIO_HOST=dremio
DREMIO_PORT=9047
DREMIO_ARROW_PORT=32010
DREMIO_USER=admin
DREMIO_PASSWORD=...
```

---

## Changes Per Existing File

### `sports_betting/fetch_top6_data.py`
- **Before**: saves CSV to `data/sports/raw/` and `data/sports/processed/`
- **After**: uploads Parquet to MinIO `fb-raw/` and `fb-processed/`
- Writes fetch status to `refresh_metadata` PostgreSQL table (replaces JSON file)

### `sports_betting/xgboost_models.py`
- **Before**: reads training data from CSV via pandas; trains on every Streamlit cold start
- **After**: training moved out of Streamlit entirely into `train_models.py` (runs as `model-trainer` service)
  - Reads feature rows from Dremio (`int_match_features` mart)
  - Saves trained `.pkl` + `metadata.json` to MinIO `fb-models/`
- `app.py` only **loads** the pre-trained pickle from MinIO at startup (no training in UI)

### `train_models.py` (NEW — standalone training job)
- Queries `int_match_features` from Dremio
- Trains `XGBClassifier` (match model + player models)
- Saves artifacts to MinIO `fb-models/xgboost_match.pkl`, `xgboost_player_*.pkl`
- Writes run metadata to PostgreSQL `refresh_metadata`
- Run order: `dbt-runner` completes → `model-trainer` runs → `streamlit` starts

### `sports_betting/generate_bet_combinations.py`
- `build_team_snapshot()` → replaced by `mart_league_standings` (dbt)
- `h2h_features_for_match()` → replaced by `int_h2h` (dbt)
- `team_last5_form()` (currently in app.py) → replaced by `int_team_form` (dbt)
- `build_context()` (currently in app.py) → replaced by `mart_match_context` (dbt)
- Core betting/odds logic (`generate_bet_combinations`, `_add_pick`) stays in Python

### `app.py`
- `load_data()` (reads CSV) → `query_dremio(sql)` (queries Dremio mart)
- `build_context()` → SQL query against `mart_match_context`
- `team_last5_form()` → SQL query against `int_team_form`
- `_cached_models` → loads pre-trained pickle from MinIO (no training in UI)
- Metadata display → reads from PostgreSQL `refresh_metadata` table
- `_fetch_data_sync()` → triggers ingestion + dbt run via API or subprocess

### `sports_betting/team_names.py`
- `TEAM_NAME_MAP` dict → becomes `dbt/seeds/team_name_map.csv`

---

## Makefile Targets

```makefile
up:          docker compose up -d
down:        docker compose down
ingest:      docker compose run --rm dbt-runner python -m sports_betting.fetch_top6_data
dbt-run:     docker compose run --rm dbt-runner dbt run --profiles-dir /dbt
dbt-test:    docker compose run --rm dbt-runner dbt test --profiles-dir /dbt
train:       docker compose run --rm model-trainer python train_models.py
migrate:     docker compose run --rm postgres psql -f /migrations/001_init.sql
logs:        docker compose logs -f streamlit
```

## PostgreSQL Migration Strategy

Migrations managed with plain SQL files (no extra tooling needed for a portfolio project):
```
migrations/
├── 001_init.sql        # Create all tables
├── 002_indexes.sql     # Add indexes on team, date columns
└── run_migrations.sh   # Applies all *.sql files in order
```
`run_migrations.sh` is called by the `postgres` service `initdb.d` entrypoint automatically on first start.

---

## Migration Phases

### Phase 1 — Infrastructure (docker-compose + MinIO)
- Write `docker-compose.yml`
- Write `sports_betting/storage.py` (MinIO client)
- Update `fetch_top6_data.py` to write Parquet → MinIO instead of CSV

### Phase 2 — PostgreSQL for config + metadata
- Write `sports_betting/db.py`
- SQL schema migrations (`migrations/001_init.sql`)
- Replace `refresh_metadata.json` with PostgreSQL table
- Move `injuries.csv` and `player_contributions.csv` to PostgreSQL

### Phase 3 — Dremio setup + connection
- Write `sports_betting/dremio.py`
- Dremio source config (MinIO + PostgreSQL) — documented in `infra/dremio_setup.md`
- Validate queries against MinIO Parquet files

### Phase 4 — dbt models
- Create `dbt/` project with staging → intermediate → mart models
- Each model replaces one Python function in generate_bet_combinations / app.py
- dbt tests for result values, null checks, date range validity

### Phase 5 — Streamlit wired to new stack
- `app.py` reads from Dremio (mart tables)
- XGBoost models trained in separate dbt/Python job, pickled to MinIO
- `app.py` loads model from MinIO at startup
- Remove direct CSV reads and pandas transformation chains

---

## What Does NOT Change

- XGBoost model architecture (still `XGBClassifier`, same features)
- Betting logic (odds calculation, combination generation, PDF export)
- UI layout and styling (Material Symbols, tabs, sidebar)
- The 15% confidence discount
- Streamlit caching strategy (`@st.cache_resource`)

---

## Folder Structure After Refactor

```
Football Bets/
├── docker-compose.yml
├── .env.example
├── Makefile
├── requirements.txt          (+ minio, psycopg2-binary, pyarrow, dremio-arrow-flight-sql-python)
├── app.py                    (updated)
├── train_models.py           (NEW — standalone training job)
├── migrations/
│   ├── 001_init.sql
│   ├── 002_indexes.sql
│   └── run_migrations.sh
├── dbt/
│   ├── Dockerfile            (dbt-core + dbt-dremio)
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── packages.yml
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_matches.sql
│   │   ├── intermediate/
│   │   │   ├── int_team_form.sql
│   │   │   ├── int_h2h.sql
│   │   │   └── int_match_features.sql
│   │   └── marts/
│   │       ├── mart_league_standings.sql
│   │       ├── mart_match_context.sql
│   │       └── mart_player_contributions.sql
│   ├── seeds/
│   │   └── team_name_map.csv
│   └── tests/
│       ├── stg_matches__result_values.sql
│       └── mart_standings__no_nulls.sql
├── infra/
│   └── dremio_setup.md       (MinIO + PostgreSQL source setup guide)
├── sports_betting/
│   ├── storage.py            (NEW — MinIO client)
│   ├── db.py                 (NEW — PostgreSQL client)
│   ├── dremio.py             (NEW — Dremio Arrow Flight query client)
│   ├── fetch_top6_data.py    (updated — writes Parquet to MinIO)
│   ├── xgboost_models.py     (updated — reads from Dremio, no training in UI)
│   ├── generate_bet_combinations.py  (simplified — betting logic only)
│   └── team_names.py         (kept during migration period)
└── data/
    └── (local CSVs retained as emergency fallback only)
```

---

## Key Dependencies to Add to `requirements.txt`

| Package                          | Purpose                              |
|----------------------------------|--------------------------------------|
| `minio`                          | MinIO Python SDK                     |
| `psycopg2-binary`                | PostgreSQL driver                    |
| `pyarrow`                        | Parquet read/write + Arrow Flight    |
| `dremio-arrow-flight-sql-python` | Dremio Arrow Flight SQL client       |
| `dbt-core`                       | dbt core (in dbt Docker image only)  |
| `dbt-dremio`                     | dbt-dremio adapter (dbt image only)  |
