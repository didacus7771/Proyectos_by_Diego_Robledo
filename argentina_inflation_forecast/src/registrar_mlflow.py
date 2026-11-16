"""Fase 5a: registro en MLflow de los experimentos de las Fases 2 a 4.

NO entrena nada. Lee los CSV deterministas que ya generaron las fases anteriores y crea un
run de MLflow por modelo y horizonte (10 modelos x 3 horizontes = 30 runs) en el experimento
"inflacion-argentina-backtest":
    parámetros  modelo, horizonte, especificación, hiperparámetros (boosting), ventana inicial,
                régimen (ventana y umbrales), n de orígenes, SHA-256 de data/raw, versiones
    métricas    MAE total y por régimen, MAE del ingenuo, diferencia contra el ingenuo con su
                intervalo de 95 %, p-valor de Diebold-Mariano, porcentaje de orígenes en que
                mejora al ingenuo, razón de error agregada en el panel (solo boosting)
    etiquetas   fase de origen, si es modelo principal, tipo de modelo
Es un registro posterior a la ejecución (decidido con Diego el 2026-11-12), no un seguimiento
durante el entrenamiento.

Almacén: base SQLite local mlflow.db y artefactos en mlruns/ dentro de la raíz del proyecto
(ambos fuera de git). Si el experimento ya tiene runs, el script se detiene con código 2 y no
duplica nada.

Uso (desde la raíz del proyecto):
    python src/registrar_mlflow.py
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scipy
import sklearn

from backtest import VENTANA_INICIAL
from modelo_global import CONFIGURACIONES, PARAMETROS_COMUNES
from regimes import UMBRAL_BAJA_MEDIA, UMBRAL_MEDIA_ALTA, VENTANA_MESES

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
CRUDOS = RAIZ / "data" / "raw"
EXPERIMENTO = "inflacion-argentina-backtest"
URI = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
ARTEFACTOS = (RAIZ / "mlruns").as_uri()

BASELINE = "ingenuo"
PRINCIPALES = ("ingenuo", "gb_base")
ORIGEN = {  # fase en que se generó cada modelo
    "ingenuo": "2", "media_3m": "2", "media_12m": "2",
    "ets_simple": "3A", "ets_amortiguado": "3A", "arima_d0": "3A", "arima_d1": "3A",
    "gb_base": "3B", "gb_simple": "3B", "gb_complejo": "3B",
}
ESPECIFICACION = {
    "ingenuo": "último valor observado",
    "media_3m": "promedio de los últimos 3 meses",
    "media_12m": "promedio de los últimos 12 meses",
    "ets_simple": "ETS(A,N,N)",
    "ets_amortiguado": "ETS(A,Ad,N)",
    "arima_d0": "ARIMA(p,0,q), p y q de 0 a 2 elegidos por AIC en cada origen",
    "arima_d1": "ARIMA(p,1,q), p y q de 0 a 2 elegidos por AIC en cada origen",
}
TIPO = {"ingenuo": "baseline", "media_3m": "baseline", "media_12m": "baseline",
        "ets_simple": "estadistico", "ets_amortiguado": "estadistico",
        "arima_d0": "estadistico", "arima_d1": "estadistico",
        "gb_base": "boosting_global", "gb_simple": "boosting_global", "gb_complejo": "boosting_global"}
REGIMENES = ("baja", "media", "alta", "total")


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_metricas() -> pd.DataFrame:
    partes = [pd.read_csv(PROC / n, usecols=["modelo", "horizonte", "regimen_origen", "n", "mae"])
              for n in ("backtest_baselines_metricas.csv", "backtest_modelos_metricas.csv",
                        "backtest_global_metricas_nacional.csv")]
    m = pd.concat(partes, ignore_index=True)
    return m.drop_duplicates(["modelo", "horizonte", "regimen_origen"], keep="first")


def construir_runs() -> list[dict]:
    met = cargar_metricas()
    res = pd.read_csv(PROC / "evaluacion_resumen.csv")
    res = res[res.regimen_origen == "total"].set_index(["modelo", "horizonte"])
    dm = pd.read_csv(PROC / "evaluacion_dm_total.csv").set_index(["modelo", "horizonte"])
    panel = pd.read_csv(PROC / "evaluacion_panel_distribucion.csv").set_index(["modelo", "horizonte"])
    datos = {f.name: sha256(f) for f in sorted(CRUDOS.glob("*.csv"))}
    entorno = {"python": platform.python_version(), "pandas": pd.__version__,
               "numpy": np.__version__, "scipy": scipy.__version__,
               "scikit_learn": sklearn.__version__, "mlflow": mlflow.__version__}
    runs = []
    for modelo, fase in ORIGEN.items():
        for h in (1, 3, 6):
            sub = met[(met.modelo == modelo) & (met.horizonte == h)].set_index("regimen_origen")
            if set(sub.index) != set(REGIMENES):
                raise SystemExit(f"Faltan métricas para {modelo} h={h}: {sorted(sub.index)}")
            params = {
                "modelo": modelo, "horizonte": h, "fase_origen": fase,
                "especificacion": ESPECIFICACION.get(modelo, "gradient boosting global"),
                "ventana_inicial_meses": VENTANA_INICIAL, "n_origenes": int(sub.loc["total", "n"]),
                "regimen_ventana_meses": VENTANA_MESES, "regimen_umbral_baja_media": UMBRAL_BAJA_MEDIA,
                "regimen_umbral_media_alta": UMBRAL_MEDIA_ALTA,
                **{f"sha256_{k}": v for k, v in datos.items()},
                **{f"version_{k}": v for k, v in entorno.items()},
            }
            if modelo in CONFIGURACIONES:
                params.update({f"gb_{k}": v for k, v in {**CONFIGURACIONES[modelo], **PARAMETROS_COMUNES}.items()})
            metricas = {f"mae_{r}": float(sub.loc[r, "mae"]) for r in REGIMENES}
            if modelo == BASELINE:
                metricas["mae_ingenuo_total"] = metricas["mae_total"]
                metricas["dif_mae_total"] = 0.0
            else:
                r = res.loc[(modelo, h)]
                metricas.update(mae_ingenuo_total=float(r.mae_ingenuo), dif_mae_total=float(r.dif_mae),
                                ic95_inf_total=float(r.ic95_inf), ic95_sup_total=float(r.ic95_sup),
                                pct_origenes_mejor_total=float(r.pct_origenes_mejor),
                                dm_p_valor_total=float(dm.loc[(modelo, h), "p_valor"]))
                if (modelo, h) in panel.index:
                    metricas["panel_razon_error_agregada"] = float(panel.loc[(modelo, h), "razon_agregada"])
                    metricas["panel_pct_series_mejor"] = float(panel.loc[(modelo, h), "pct_series_mejor"])
            runs.append(dict(nombre=f"{modelo}_h{h}", params=params, metricas=metricas,
                             etiquetas={"fase": fase, "tipo": TIPO[modelo],
                                        "principal": str(modelo in PRINCIPALES)}))
    return runs


def main() -> int:
    mlflow.set_tracking_uri(URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENTO)
    if exp is None:
        exp_id = mlflow.create_experiment(EXPERIMENTO, artifact_location=ARTEFACTOS)
    else:
        exp_id = exp.experiment_id
        existentes = mlflow.search_runs(experiment_ids=[exp_id], filter_string="status = 'FINISHED'")
        if len(existentes) > 0:
            print(f"El experimento '{EXPERIMENTO}' ya tiene {len(existentes)} runs; no se duplica nada.")
            return 2
    runs = construir_runs()
    for r in runs:
        with mlflow.start_run(experiment_id=exp_id, run_name=r["nombre"]):
            mlflow.log_params(r["params"])
            mlflow.log_metrics(r["metricas"])
            mlflow.set_tags(r["etiquetas"])
    print(f"Registro en MLflow: {len(runs)} runs en '{EXPERIMENTO}' ({URI})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
