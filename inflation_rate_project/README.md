# Global inflation ETL: Python, PostgreSQL, Airbyte and Snowflake

An end-to-end pipeline built to test how **Postgres, Docker, Airbyte and Snowflake** work together. It starts from a CSV with the inflation of 196 countries since 1980, cleans and reshapes it with pandas, loads it into PostgreSQL running in Docker, and syncs it to Snowflake with Airbyte.

```mermaid
flowchart LR
    A[CSV<br/>196 countries x 47 columns] --> B[pandas<br/>melt + continent column]
    B --> C[CSV by continent<br/>+ inflation_data.csv]
    C --> D[PostgreSQL<br/>on Docker]
    D --> E[Airbyte]
    E --> F[Snowflake]
```

## Pipeline

### 1. Extract and explore
- Loads `global_inflation_data.csv` with pandas: 196 rows and 47 columns (45 `float64`, 2 `object`).
- Checks types, shape and null values per column.
- **Cleaning decision:** the nulls are concentrated before the year 2000. Some reflect real history (for example Ukraine, which becomes an independent state after the dissolution of the Soviet Union) and others missing reports from countries. I chose not to impute them.

### 2. Transform
- `pd.melt()` turns the year columns into rows: `Country`, `Indicator`, `Year`, `Inflation`.
- Countries are grouped by continent with lists and `isin()`, and a `Continent` column is added to each table.
- The five tables are saved as CSV files in `csv/` and combined into `inflation_data.csv`.

### 3. Load to PostgreSQL (Docker)
- `postgres_docker.yml` starts a PostgreSQL container with a persistent volume.
- Two databases are created: `INFLATION_GLOBAL` (one table with all the data) and `INFLATION_CONTINENT` (one table per continent), loaded with `\COPY`.

### 4. Sync to Snowflake (Airbyte)
- Snowflake is prepared with the setup script that Airbyte provides (role, user, warehouse, database and schema).
- Airbyte runs locally (`bash run-ab-platform.sh`) with PostgreSQL as source and Snowflake as destination.
- The final check is in Snowflake, where the `INFL_GLOBAL` table appears with the full dataset.

## Files

| File | Purpose |
|---|---|
| `inflation_etl.ipynb` | Extraction, exploration and transformation notebook |
| `postgres_docker.yml` | Docker Compose file for PostgreSQL |
| `global_inflation_data.csv` | Source data |
| `csv/` | Output tables by continent and the combined file |
| `proyecto_inflacion_global.docx` | Full step-by-step walkthrough with screenshots |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook inflation_etl.ipynb      # edit the CSV path at the top

# set your own POSTGRES_PASSWORD and the volume path in the file first
docker compose -f postgres_docker.yml up -d
```

Then follow the walkthrough in `proyecto_inflacion_global.docx` for the Snowflake and Airbyte steps.

## Tools

Python, pandas, Docker, PostgreSQL, Airbyte, Snowflake, Tableau.
