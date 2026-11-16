"""Fase 5a: carga de los resultados en PostgreSQL (Docker) para el dashboard y el análisis en SQL.

NO recalcula nada. Lee los CSV de data/processed/ y los carga en el esquema "inflacion" de la
base definida en .env (ver .env.example y docker-compose.yml). La carga es completa e
idempotente: sql/01_esquema.sql borra y recrea el esquema, luego se cargan las tablas y se
crean las vistas de sql/02_vistas.sql. Al final compara las filas de cada tabla con las de su
CSV y termina con código 1 si alguna no coincide.

Credenciales: solo desde .env o variables de entorno (POSTGRES_USER, POSTGRES_PASSWORD,
POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT). El script nunca imprime la contraseña.

Uso (desde la raíz del proyecto, con la base levantada):
    python src/cargar_postgres.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
SQL = RAIZ / "sql"
ESQUEMA = "inflacion"
OBLIGATORIAS = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")

# modelo -> (fase de origen, tipo, principal)
MODELOS = {
    "ingenuo": ("2", "baseline", True), "media_3m": ("2", "baseline", False),
    "media_12m": ("2", "baseline", False),
    "ets_simple": ("3A", "estadistico", False), "ets_amortiguado": ("3A", "estadistico", False),
    "arima_d0": ("3A", "estadistico", False), "arima_d1": ("3A", "estadistico", False),
    "gb_base": ("3B", "boosting_global", True), "gb_simple": ("3B", "boosting_global", False),
    "gb_complejo": ("3B", "boosting_global", False),
}
COLS_PRON = ["origen", "horizonte", "fecha_objetivo", "modelo", "y_real", "y_pred", "error",
             "error_abs", "regimen_origen"]
COLS_MET = ["modelo", "horizonte", "regimen_origen", "n", "mae", "mediana_error_abs", "p90_error_abs",
            "desvio_error_abs", "sesgo", "mae_ingenuo", "mae_relativo"]
COLS_SERIE = ["fecha", "serie_id", "codigo", "descripcion", "clasificador", "region", "indice",
              "var_mensual", "var_interanual"]


def leer_env(ruta: Path) -> dict[str, str]:
    valores: dict[str, str] = {}
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if linea and not linea.startswith("#") and "=" in linea:
                clave, _, valor = linea.partition("=")
                valores[clave.strip()] = valor.strip().strip('"').strip("'")
    return valores


def conexion() -> URL:
    cfg = {**leer_env(RAIZ / ".env"), **{k: v for k, v in os.environ.items() if k.startswith("POSTGRES_")}}
    faltan = [k for k in OBLIGATORIAS if not cfg.get(k)]
    if faltan:
        raise SystemExit(f"Faltan variables en .env o en el entorno: {', '.join(faltan)}")
    return URL.create("postgresql+psycopg2", username=cfg["POSTGRES_USER"], password=cfg["POSTGRES_PASSWORD"],
                      host=cfg.get("POSTGRES_HOST", "localhost"), port=int(cfg.get("POSTGRES_PORT", "5433")),
                      database=cfg["POSTGRES_DB"])


def tablas() -> dict[str, pd.DataFrame]:
    pron = pd.concat([pd.read_csv(PROC / n, usecols=COLS_PRON) for n in (
        "backtest_baselines_pronosticos.csv", "backtest_modelos_pronosticos.csv",
        "backtest_global_pronosticos_nacional.csv")], ignore_index=True)
    met = pd.concat([pd.read_csv(PROC / n, usecols=COLS_MET) for n in (
        "backtest_baselines_metricas.csv", "backtest_modelos_metricas.csv",
        "backtest_global_metricas_nacional.csv")], ignore_index=True)
    met = met.drop_duplicates(["modelo", "horizonte", "regimen_origen"], keep="first")
    for col in ("origen", "fecha_objetivo"):
        pron[col] = pd.to_datetime(pron[col]).dt.date
    serie = pd.read_csv(PROC / "ipc_divisiones_tidy.csv", usecols=COLS_SERIE, dtype={"codigo": str})
    serie["fecha"] = pd.to_datetime(serie["fecha"]).dt.date
    modelos = pd.DataFrame([(m, *v) for m, v in MODELOS.items()],
                           columns=["modelo", "fase_origen", "tipo", "principal"])
    return {
        "modelos": modelos,
        "serie_ipc": serie,
        "pronosticos_nacional": pron,
        "metricas_nacional": met,
        "metricas_series": pd.read_csv(PROC / "backtest_global_metricas_series.csv"),
        "evaluacion_resumen": pd.read_csv(PROC / "evaluacion_resumen.csv"),
        "evaluacion_dm_total": pd.read_csv(PROC / "evaluacion_dm_total.csv"),
        "evaluacion_sensibilidad": pd.read_csv(PROC / "evaluacion_sensibilidad.csv"),
        "evaluacion_cuantiles": pd.read_csv(PROC / "evaluacion_cuantiles.csv"),
        "evaluacion_panel_distribucion": pd.read_csv(PROC / "evaluacion_panel_distribucion.csv"),
    }


def main() -> int:
    datos = tablas()
    motor = create_engine(conexion())
    with motor.begin() as con:
        con.execute(text((SQL / "01_esquema.sql").read_text(encoding="utf-8")))
        for nombre, df in datos.items():
            df.to_sql(nombre, con, schema=ESQUEMA, if_exists="append", index=False,
                      chunksize=2000, method="multi")
        con.execute(text((SQL / "02_vistas.sql").read_text(encoding="utf-8")))
    print(f"Carga en PostgreSQL, esquema '{ESQUEMA}':")
    fallos = 0
    with motor.connect() as con:
        for nombre, df in datos.items():
            n_db = con.execute(text(f"SELECT count(*) FROM {ESQUEMA}.{nombre}")).scalar_one()
            ok = n_db == len(df)
            fallos += not ok
            print(f"  {nombre:<32} csv {len(df):>6} | base {n_db:>6} | {'OK' if ok else 'DIFIERE'}")
        for vista in ("vw_evaluacion", "vw_pronosticos", "vw_error_ingenuo"):
            n_v = con.execute(text(f"SELECT count(*) FROM {ESQUEMA}.{vista}")).scalar_one()
            print(f"  {vista:<32} filas {n_v}")
    print("Resultado:", "OK" if fallos == 0 else f"{fallos} tabla(s) con filas distintas")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
