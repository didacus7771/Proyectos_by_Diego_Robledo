"""Fase 2: baselines y backtesting de origen móvil.

Pronostica la variación mensual del nivel general nacional (IPC INDEC) a 1, 3 y 6
meses con tres métodos simples y mide el error en puntos porcentuales (pp).
Lee data/processed/ipc_divisiones_tidy.csv (salida de ingest.py) y escribe dos
archivos en data/processed/:
    backtest_baselines_pronosticos.csv   un pronóstico por fila
    backtest_baselines_metricas.csv      MAE y dispersión por método, horizonte y régimen

Definiciones (fijas, no se cambian después de ver resultados):
    - Objetivo: la variación mensual del mes (origen + h), no la acumulada.
    - Origen: último mes observado. El primer origen es el mes 36 de la serie
      (ventana inicial de 36 observaciones, que crece en cada origen).
    - Régimen de un pronóstico: el vigente en su fecha de origen (src/regimes.py).
    - Baseline oficial del proyecto: "ingenuo" (último valor observado).

Uso (desde la raíz del proyecto):
    python src/backtest.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regimes import ETIQUETAS, regimen

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA_POR_DEFECTO = RAIZ / "data" / "processed" / "ipc_divisiones_tidy.csv"
SALIDA_PRONOSTICOS = RAIZ / "data" / "processed" / "backtest_baselines_pronosticos.csv"
SALIDA_METRICAS = RAIZ / "data" / "processed" / "backtest_baselines_metricas.csv"

CODIGO_NIVEL_GENERAL = "0"
REGION = "Nacional"
HORIZONTES = (1, 3, 6)
VENTANA_INICIAL = 36
BASELINE_OFICIAL = "ingenuo"


def _ingenuo(historia: np.ndarray) -> float:
    """Último valor observado."""
    return float(historia[-1])


def _media_3m(historia: np.ndarray) -> float:
    """Promedio de los últimos 3 meses."""
    return float(historia[-3:].mean())


def _media_12m(historia: np.ndarray) -> float:
    """Promedio de los últimos 12 meses."""
    return float(historia[-12:].mean())


BASELINES = {
    "ingenuo": _ingenuo,
    "media_3m": _media_3m,
    "media_12m": _media_12m,
}


def cargar_serie(ruta: Path = ENTRADA_POR_DEFECTO) -> pd.Series:
    """Variación mensual (%) del nivel general nacional, indexada por fecha."""
    tidy = pd.read_csv(ruta, parse_dates=["fecha"], dtype={"codigo": str})
    filtro = (tidy["codigo"] == CODIGO_NIVEL_GENERAL) & (tidy["region"] == REGION)
    serie = tidy.loc[filtro].set_index("fecha")["var_mensual"].sort_index().dropna()
    serie.name = "var_mensual"
    return serie


def backtest(serie: pd.Series) -> pd.DataFrame:
    """Backtesting de origen móvil con ventana creciente.

    En cada origen, cada baseline ve solo la historia hasta esa fecha (inclusive).
    """
    regimenes = regimen(serie)
    n = len(serie)
    filas = []
    for h in HORIZONTES:
        for i in range(VENTANA_INICIAL - 1, n - h):
            historia = serie.iloc[: i + 1].to_numpy()
            origen = serie.index[i]
            objetivo = serie.index[i + h]
            y_real = float(serie.iloc[i + h])
            reg = regimenes.iloc[i]
            for nombre, funcion in BASELINES.items():
                y_pred = funcion(historia)
                filas.append(
                    {
                        "origen": origen,
                        "horizonte": h,
                        "fecha_objetivo": objetivo,
                        "modelo": nombre,
                        "y_real": y_real,
                        "y_pred": y_pred,
                        "error": y_pred - y_real,
                        "error_abs": abs(y_pred - y_real),
                        "regimen_origen": reg,
                    }
                )
    return pd.DataFrame(filas)


def metricas(pronosticos: pd.DataFrame) -> pd.DataFrame:
    """MAE y dispersión por modelo, horizonte y régimen (más una fila 'total')."""
    p = pronosticos.copy()
    p["regimen_origen"] = p["regimen_origen"].astype(str)
    total = p.assign(regimen_origen="total")
    todo = pd.concat([p, total], ignore_index=True)

    def resumir(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(g),
                "mae": g["error_abs"].mean(),
                "mediana_error_abs": g["error_abs"].median(),
                "p90_error_abs": g["error_abs"].quantile(0.9),
                "desvio_error_abs": g["error_abs"].std(ddof=1),
                "sesgo": g["error"].mean(),
            }
        )

    claves = ["modelo", "horizonte", "regimen_origen"]
    res = todo.groupby(claves, observed=True).apply(resumir, include_groups=False).reset_index()
    ref = (
        res[res["modelo"] == BASELINE_OFICIAL]
        .set_index(["horizonte", "regimen_origen"])["mae"]
        .rename("mae_ingenuo")
    )
    res = res.join(ref, on=["horizonte", "regimen_origen"])
    res["mae_relativo"] = res["mae"] / res["mae_ingenuo"]
    orden_reg = {"baja": 0, "media": 1, "alta": 2, "total": 3}
    res["_o"] = res["regimen_origen"].map(orden_reg)
    res = res.sort_values(["horizonte", "modelo", "_o"]).drop(columns="_o")
    res["n"] = res["n"].astype(int)
    return res.reset_index(drop=True)


def validar(serie: pd.Series, pronosticos: pd.DataFrame) -> dict:
    """Comprobaciones de integridad. Lanza AssertionError si algo falla."""
    n = len(serie)
    resumen = {"observaciones": n}
    for h in HORIZONTES:
        esperados = n - VENTANA_INICIAL - h + 1
        obtenidos = pronosticos[(pronosticos["horizonte"] == h) & (pronosticos["modelo"] == BASELINE_OFICIAL)].shape[0]
        assert obtenidos == esperados, f"h={h}: {obtenidos} orígenes, se esperaban {esperados}"
        resumen[f"origenes_h{h}"] = obtenidos

    assert not pronosticos[["y_real", "y_pred", "error_abs"]].isna().any().any(), "hay valores nulos"
    assert pronosticos["regimen_origen"].notna().all(), "hay orígenes sin régimen"
    assert (pronosticos["fecha_objetivo"] > pronosticos["origen"]).all(), "objetivo no posterior al origen"

    # El objetivo está exactamente h meses después del origen.
    meses = (
        (pronosticos["fecha_objetivo"].dt.year - pronosticos["origen"].dt.year) * 12
        + (pronosticos["fecha_objetivo"].dt.month - pronosticos["origen"].dt.month)
    )
    assert (meses == pronosticos["horizonte"]).all(), "salto de meses distinto del horizonte"

    # Sin fuga: el ingenuo coincide con el valor observado en la fecha de origen.
    ing = pronosticos[pronosticos["modelo"] == BASELINE_OFICIAL]
    assert np.allclose(ing["y_pred"].to_numpy(), serie.loc[ing["origen"]].to_numpy()), "el ingenuo no usa el valor del origen"

    # Sin fuga: el régimen de cada origen se reproduce usando solo la historia hasta esa fecha.
    for origen in pronosticos["origen"].drop_duplicates():
        esperado = regimen(serie.loc[:origen]).iloc[-1]
        obtenido = pronosticos.loc[pronosticos["origen"] == origen, "regimen_origen"].iloc[0]
        assert esperado == obtenido, f"régimen distinto en el origen {origen:%Y-%m}"

    primer = pronosticos["origen"].min()
    assert serie.index.get_loc(primer) == VENTANA_INICIAL - 1, "el primer origen no es el mes 36"
    resumen["primer_origen"] = f"{primer:%Y-%m}"
    resumen["ultimo_objetivo"] = f"{pronosticos['fecha_objetivo'].max():%Y-%m}"
    return resumen


def main(entrada: Path = ENTRADA_POR_DEFECTO) -> pd.DataFrame:
    entrada = Path(entrada)
    if not entrada.exists():
        raise FileNotFoundError(f"No se encontró {entrada}. Ejecuta antes: python src/ingest.py")
    serie = cargar_serie(entrada)
    pronosticos = backtest(serie)
    resumen = validar(serie, pronosticos)
    tabla = metricas(pronosticos)

    SALIDA_PRONOSTICOS.parent.mkdir(parents=True, exist_ok=True)
    pronosticos.assign(regimen_origen=pronosticos["regimen_origen"].astype(str)).to_csv(
        SALIDA_PRONOSTICOS, index=False, encoding="utf-8"
    )
    tabla.to_csv(SALIDA_METRICAS, index=False, encoding="utf-8")

    print(f"Serie: nivel general, {REGION} -> {resumen['observaciones']} observaciones")
    for clave, valor in resumen.items():
        if clave != "observaciones":
            print(f"  {clave}: {valor}")
    print(f"Pronósticos: {len(pronosticos):,} filas -> {SALIDA_PRONOSTICOS.name}")
    print(f"Métricas: {len(tabla)} filas -> {SALIDA_METRICAS.name}")
    print()
    vista = tabla[["horizonte", "modelo", "regimen_origen", "n", "mae", "mae_relativo"]].copy()
    vista["mae"] = vista["mae"].round(3)
    vista["mae_relativo"] = vista["mae_relativo"].round(3)
    print(vista.to_string(index=False))
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--entrada", type=Path, default=ENTRADA_POR_DEFECTO)
    args = parser.parse_args()
    main(args.entrada)
