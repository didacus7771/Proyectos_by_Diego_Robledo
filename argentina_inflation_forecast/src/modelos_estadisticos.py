"""Fase 3A: modelos estadísticos por serie (ETS y ARIMA) con backtesting de origen móvil.

Pronostica la variación mensual del nivel general nacional (IPC INDEC) a 1, 3 y 6 meses
con los mismos orígenes, horizontes, objetivo y regímenes que los baselines de la Fase 2
(src/backtest.py), y los compara contra ellos.

Modelos (especificaciones fijas, definidas antes de ver resultados):
    ets_simple        ETS aditivo sin tendencia ni estacionalidad.
    ets_amortiguado   ETS aditivo con tendencia amortiguada, sin estacionalidad.
    arima_d0          ARIMA(p,0,q) con constante; p y q en 0..2 elegidos por AIC en cada origen.
    arima_d1          ARIMA(p,1,q) sin constante; p y q en 0..2 elegidos por AIC en cada origen.
No se elige d por modelo: los dos casos se reportan porque la prueba ADF de la Fase 1 fue
dudosa (p = 0,074). No hay estacionalidad: la autocorrelación en el rezago 12 es 0,12.

Reglas:
    - Cada ajuste usa solo la serie hasta la fecha de origen (inclusive). Se ajusta una vez
      por origen y por modelo y se toman los pasos 1, 3 y 6 del mismo ajuste.
    - Si un ajuste falla, el pronóstico de ese modelo en ese origen es el del ingenuo y la
      fila queda marcada con fallo=True. Los fallos y las advertencias se cuentan y se informan.

Uso (desde la raíz del proyecto):
    python src/modelos_estadisticos.py
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from backtest import (
    BASELINE_OFICIAL,
    ENTRADA_POR_DEFECTO,
    HORIZONTES,
    VENTANA_INICIAL,
    backtest as backtest_baselines,
    cargar_serie,
    metricas,
)
from regimes import regimen

RAIZ = Path(__file__).resolve().parents[1]
SALIDA_PRONOSTICOS = RAIZ / "data" / "processed" / "backtest_modelos_pronosticos.csv"
SALIDA_METRICAS = RAIZ / "data" / "processed" / "backtest_modelos_metricas.csv"

MODELOS = ("ets_simple", "ets_amortiguado", "arima_d0", "arima_d1")
ORDEN_MAXIMO = 2  # p y q de 0 a 2
PASOS = max(HORIZONTES)
UMBRAL_FALLOS = 0.10  # validar() falla si más del 10 % de los ajustes cae al ingenuo


def _ajustar_ets(historia: pd.Series, amortiguado: bool) -> tuple[np.ndarray, str]:
    modelo = ETSModel(
        historia,
        error="add",
        trend="add" if amortiguado else None,
        damped_trend=amortiguado,
    )
    ajuste = modelo.fit(disp=False)
    nombre = "ETS(A,Ad,N)" if amortiguado else "ETS(A,N,N)"
    return np.asarray(ajuste.forecast(PASOS), dtype=float), nombre


def _ajustar_arima(historia: pd.Series, d: int) -> tuple[np.ndarray, str]:
    """Elige (p, q) por AIC entre 0..ORDEN_MAXIMO con d fijo; devuelve el pronóstico."""
    tendencia = "c" if d == 0 else "n"
    mejor = None
    for p, q in itertools.product(range(ORDEN_MAXIMO + 1), repeat=2):
        try:
            ajuste = ARIMA(historia, order=(p, d, q), trend=tendencia).fit()
        except Exception:
            continue
        if not np.isfinite(ajuste.aic):
            continue
        if mejor is None or ajuste.aic < mejor[0]:
            mejor = (ajuste.aic, p, q, ajuste)
    if mejor is None:
        raise RuntimeError(f"ningún ARIMA(p,{d},q) convergió")
    _, p, q, ajuste = mejor
    return np.asarray(ajuste.forecast(PASOS), dtype=float), f"ARIMA({p},{d},{q})"


def _pronosticar(nombre: str, historia: pd.Series) -> tuple[np.ndarray, str]:
    if nombre == "ets_simple":
        return _ajustar_ets(historia, amortiguado=False)
    if nombre == "ets_amortiguado":
        return _ajustar_ets(historia, amortiguado=True)
    if nombre == "arima_d0":
        return _ajustar_arima(historia, d=0)
    if nombre == "arima_d1":
        return _ajustar_arima(historia, d=1)
    raise ValueError(nombre)


def backtest_modelos(serie: pd.Series) -> pd.DataFrame:
    """Un ajuste por origen y por modelo; se usan los pasos 1, 3 y 6 del mismo ajuste."""
    regimenes = regimen(serie)
    n = len(serie)
    maximo_h = min(HORIZONTES)  # el origen más tardío es el del horizonte más corto
    filas = []
    for i in range(VENTANA_INICIAL - 1, n - maximo_h):
        historia = serie.iloc[: i + 1]
        origen = serie.index[i]
        reg = regimenes.iloc[i]
        for nombre in MODELOS:
            fallo, especificacion, n_advertencias = False, "", 0
            with warnings.catch_warnings(record=True) as capturadas:
                warnings.simplefilter("always")
                try:
                    pasos, especificacion = _pronosticar(nombre, historia)
                    if not np.all(np.isfinite(pasos)):
                        raise ValueError("pronóstico no finito")
                except Exception as error:  # el ajuste falla: se usa el ingenuo y se marca
                    fallo = True
                    especificacion = f"fallo: {type(error).__name__}"
                    pasos = np.full(PASOS, float(historia.iloc[-1]))
            n_advertencias = len(capturadas)
            for h in HORIZONTES:
                if i + h >= n:
                    continue
                y_real = float(serie.iloc[i + h])
                y_pred = float(pasos[h - 1])
                filas.append(
                    {
                        "origen": origen,
                        "horizonte": h,
                        "fecha_objetivo": serie.index[i + h],
                        "modelo": nombre,
                        "y_real": y_real,
                        "y_pred": y_pred,
                        "error": y_pred - y_real,
                        "error_abs": abs(y_pred - y_real),
                        "regimen_origen": reg,
                        "ultima_fecha_train": historia.index[-1],
                        "especificacion": especificacion,
                        "fallo": fallo,
                        "n_advertencias": n_advertencias,
                    }
                )
    return pd.DataFrame(filas)


def validar(serie: pd.Series, pron: pd.DataFrame) -> dict:
    """Comprobaciones de integridad. Lanza AssertionError si algo falla."""
    n = len(serie)
    resumen: dict = {}
    for nombre in MODELOS:
        for h in HORIZONTES:
            esperados = n - VENTANA_INICIAL - h + 1
            obtenidos = ((pron["modelo"] == nombre) & (pron["horizonte"] == h)).sum()
            assert obtenidos == esperados, f"{nombre} h={h}: {obtenidos} filas, se esperaban {esperados}"
    assert not pron[["y_real", "y_pred", "error_abs"]].isna().any().any(), "hay valores nulos"
    assert np.isfinite(pron["y_pred"]).all(), "hay pronósticos no finitos"
    assert (pron["ultima_fecha_train"] == pron["origen"]).all(), "un ajuste usó datos posteriores al origen"
    assert (pron["fecha_objetivo"] > pron["origen"]).all(), "objetivo no posterior al origen"
    meses = (
        (pron["fecha_objetivo"].dt.year - pron["origen"].dt.year) * 12
        + (pron["fecha_objetivo"].dt.month - pron["origen"].dt.month)
    )
    assert (meses == pron["horizonte"]).all(), "salto de meses distinto del horizonte"
    for origen in pron["origen"].drop_duplicates():
        esperado = regimen(serie.loc[:origen]).iloc[-1]
        obtenido = pron.loc[pron["origen"] == origen, "regimen_origen"].iloc[0]
        assert esperado == obtenido, f"régimen distinto en el origen {origen:%Y-%m}"
    ajustes = pron.drop_duplicates(["origen", "modelo"])
    tasa = ajustes["fallo"].mean()
    assert tasa <= UMBRAL_FALLOS, f"{tasa:.1%} de ajustes fallaron"
    resumen["ajustes"] = len(ajustes)
    resumen["ajustes_con_fallo"] = int(ajustes["fallo"].sum())
    resumen["ajustes_con_advertencias"] = int((ajustes["n_advertencias"] > 0).sum())
    return resumen


def main(entrada: Path = ENTRADA_POR_DEFECTO) -> pd.DataFrame:
    entrada = Path(entrada)
    if not entrada.exists():
        raise FileNotFoundError(f"No se encontró {entrada}. Ejecuta antes: python src/ingest.py")
    serie = cargar_serie(entrada).asfreq("MS")
    assert not serie.isna().any(), "la serie tiene meses faltantes"

    inicio = time.time()
    pron = backtest_modelos(serie)
    resumen = validar(serie, pron)
    minutos = (time.time() - inicio) / 60

    base = backtest_baselines(serie)
    todos = pd.concat(
        [base.assign(regimen_origen=base["regimen_origen"].astype(str)),
         pron.assign(regimen_origen=pron["regimen_origen"].astype(str))],
        ignore_index=True,
    )
    tabla = metricas(todos)

    SALIDA_PRONOSTICOS.parent.mkdir(parents=True, exist_ok=True)
    pron.assign(regimen_origen=pron["regimen_origen"].astype(str)).to_csv(
        SALIDA_PRONOSTICOS, index=False, encoding="utf-8"
    )
    tabla.to_csv(SALIDA_METRICAS, index=False, encoding="utf-8")

    print(f"Serie: nivel general, Nacional -> {len(serie)} observaciones")
    print(f"Modelos: {', '.join(MODELOS)}")
    for clave, valor in resumen.items():
        print(f"  {clave}: {valor}")
    print(f"Pronósticos: {len(pron):,} filas -> {SALIDA_PRONOSTICOS.name}")
    print(f"Métricas (baselines + modelos): {len(tabla)} filas -> {SALIDA_METRICAS.name}")
    print()
    vista = tabla[tabla["regimen_origen"] == "total"][["horizonte", "modelo", "n", "mae", "mae_relativo"]].copy()
    vista[["mae", "mae_relativo"]] = vista[["mae", "mae_relativo"]].round(3)
    print("MAE total por horizonte (pp):")
    print(vista.to_string(index=False))
    print()
    por_reg = tabla[tabla["regimen_origen"] != "total"][["horizonte", "modelo", "regimen_origen", "n", "mae", "mae_relativo"]].copy()
    por_reg[["mae", "mae_relativo"]] = por_reg[["mae", "mae_relativo"]].round(3)
    print("MAE por régimen (pp):")
    print(por_reg.to_string(index=False))
    print(f"\nTiempo de ajuste: {minutos:.1f} min")
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--entrada", type=Path, default=ENTRADA_POR_DEFECTO)
    args = parser.parse_args()
    main(args.entrada)
