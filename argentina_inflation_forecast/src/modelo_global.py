"""Fase 3B: modelo global de gradient boosting sobre el panel de series (IPC INDEC).

Un solo modelo por horizonte aprende de las 126 series (18 códigos x 7 regiones) a la vez
y pronostica la variación mensual a 1, 3 y 6 meses. Usa los mismos orígenes, horizontes y
regímenes que los baselines (src/backtest.py) y los modelos estadísticos
(src/modelos_estadisticos.py).

Diseño fijo, definido antes de ver resultados:
    Objetivo       Se aprende la DIFERENCIA y(u+h) - y(u) y el pronóstico es y(origen) + diferencia.
                   Así el ingenuo es el ancla y los árboles no necesitan extrapolar niveles
                   nunca vistos (la inflación llegó a 25,5 % con una historia previa menor).
    Pérdida        Error absoluto (coincide con la métrica MAE y resiste valores extremos).
    Variables      Solo información disponible en la fecha u: nivel actual de la serie, sus
                   diferencias con 1, 2, 3 y 6 meses atrás, promedios móviles de 3 y 12
                   meses, nivel y promedio de 3 meses del nivel general nacional, régimen
                   nacional vigente (0 baja, 1 media, 2 alta), código de la serie y región.
    Entrenamiento  En cada origen, solo muestras cuyo objetivo ya era observable (u + h <= origen),
                   desde u = 11 (hace falta el promedio de 12 meses). Se reentrena en cada origen.
    Configuraciones Tres, fijadas de antemano y reportadas TODAS; la principal es gb_base.
                   Ninguna se elige después de ver resultados.
                       gb_base      profundidad 3, 200 iteraciones, tasa 0,05
                       gb_simple    profundidad 2, 100 iteraciones, tasa 0,05
                       gb_complejo  profundidad 5, 300 iteraciones, tasa 0,05
Salidas en data/processed/:
    backtest_global_pronosticos_nacional.csv   pronósticos del nivel general nacional
    backtest_global_metricas_nacional.csv      MAE por modelo, horizonte y régimen (con el ingenuo)
    backtest_global_metricas_series.csv        MAE relativo al ingenuo por serie y horizonte
    backtest_global_metricas_panel.csv         resumen del panel por modelo, horizonte y régimen
    backtest_global_salida.txt                 copia (UTF-8) de lo que imprime el script

Uso (desde la raíz del proyecto):
    python src/modelo_global.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from backtest import ENTRADA_POR_DEFECTO, HORIZONTES, VENTANA_INICIAL, metricas
from regimes import ETIQUETAS, regimen

RAIZ = Path(__file__).resolve().parents[1]
SALIDA_PRON_NAC = RAIZ / "data" / "processed" / "backtest_global_pronosticos_nacional.csv"
SALIDA_MET_NAC = RAIZ / "data" / "processed" / "backtest_global_metricas_nacional.csv"
SALIDA_MET_SERIES = RAIZ / "data" / "processed" / "backtest_global_metricas_series.csv"
SALIDA_MET_PANEL = RAIZ / "data" / "processed" / "backtest_global_metricas_panel.csv"
SALIDA_REGISTRO = RAIZ / "data" / "processed" / "backtest_global_salida.txt"

SERIE_NACIONAL = "0|Nacional"
PRIMER_ENTRENAMIENTO = 11  # índice del primer mes con promedio de 12 meses completo
SEMILLA = 0
CONFIGURACIONES = {
    "gb_base": dict(max_depth=3, max_iter=200, learning_rate=0.05),
    "gb_simple": dict(max_depth=2, max_iter=100, learning_rate=0.05),
    "gb_complejo": dict(max_depth=5, max_iter=300, learning_rate=0.05),
}
PARAMETROS_COMUNES = dict(
    loss="absolute_error",
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=SEMILLA,
)
NOMBRES_VARIABLES = [
    "nivel", "dif1", "dif2", "dif3", "dif6", "media3", "media12",
    "nac_nivel", "nac_media3", "nac_regimen", "codigo", "region",
]
INDICES_CATEGORICAS = [NOMBRES_VARIABLES.index("codigo"), NOMBRES_VARIABLES.index("region")]


def cargar_panel(ruta: Path = ENTRADA_POR_DEFECTO) -> pd.DataFrame:
    """Variación mensual (%) de las 126 series: filas = meses (2017-01 a 2026-09), columnas = serie_id."""
    tidy = pd.read_csv(ruta, parse_dates=["fecha"], dtype={"codigo": str})
    ancho = tidy.pivot(index="fecha", columns="serie_id", values="var_mensual").sort_index()
    ancho = ancho.dropna(how="all")  # el primer mes (2016-12) no tiene variación mensual
    ancho = ancho.asfreq("MS")
    assert not ancho.isna().any().any(), "el panel tiene valores faltantes"
    assert SERIE_NACIONAL in ancho.columns, "falta la serie nacional"
    return ancho


def construir_variables(panel: pd.DataFrame) -> np.ndarray:
    """Matriz (meses, series, variables). Cada fila usa solo datos hasta su propio mes."""
    t, s = panel.shape
    nac = panel[SERIE_NACIONAL]
    codigos = pd.Series([c.split("|")[0] for c in panel.columns])
    regiones = pd.Series([c.split("|")[1] for c in panel.columns])
    cod_num = codigos.astype("category").cat.codes.to_numpy()
    reg_num = regiones.astype("category").cat.codes.to_numpy()
    reg_nac = regimen(nac).cat.codes.to_numpy().astype(float)
    reg_nac[reg_nac < 0] = np.nan

    def en_matriz(vector: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(vector, dtype=float)[:, None], s, axis=1)

    nivel = panel.to_numpy(dtype=float)
    variables = [
        nivel,
        nivel - panel.shift(1).to_numpy(),
        nivel - panel.shift(2).to_numpy(),
        nivel - panel.shift(3).to_numpy(),
        nivel - panel.shift(6).to_numpy(),
        panel.rolling(3).mean().to_numpy(),
        panel.rolling(12).mean().to_numpy(),
        en_matriz(nac.to_numpy()),
        en_matriz(nac.rolling(3).mean().to_numpy()),
        en_matriz(reg_nac),
        np.repeat(cod_num[None, :], t, axis=0).astype(float),
        np.repeat(reg_num[None, :], t, axis=0).astype(float),
    ]
    return np.stack(variables, axis=2)


def _modelo(nombre: str) -> HistGradientBoostingRegressor:
    mascara = np.zeros(len(NOMBRES_VARIABLES), dtype=bool)
    mascara[INDICES_CATEGORICAS] = True
    return HistGradientBoostingRegressor(
        **CONFIGURACIONES[nombre], **PARAMETROS_COMUNES, categorical_features=mascara
    )


def backtest_global(panel: pd.DataFrame) -> pd.DataFrame:
    """Pronósticos de todas las series en cada origen. Devuelve formato largo."""
    nivel = panel.to_numpy(dtype=float)
    t, s = nivel.shape
    x = construir_variables(panel)
    fechas = panel.index
    series = np.array(panel.columns)
    reg_nac = regimen(panel[SERIE_NACIONAL])
    filas = []
    inicio = time.time()
    for h in HORIZONTES:
        for i in range(VENTANA_INICIAL - 1, t - h):
            u = np.arange(PRIMER_ENTRENAMIENTO, i - h + 1)  # el objetivo de u es u + h <= i
            x_train = x[u].reshape(-1, x.shape[2])
            y_train = (nivel[u + h] - nivel[u]).reshape(-1)
            x_test = x[i]
            y_real = nivel[i + h]
            ingenuo = nivel[i]
            base = {
                "origen": fechas[i], "horizonte": h, "fecha_objetivo": fechas[i + h],
                "regimen_origen": reg_nac.iloc[i], "n_train": len(y_train),
                "fecha_max_objetivo_train": fechas[u.max() + h],
            }
            preds = {"ingenuo": ingenuo}
            for nombre in CONFIGURACIONES:
                modelo = _modelo(nombre).fit(x_train, y_train)
                preds[nombre] = ingenuo + modelo.predict(x_test)
            for nombre, y_pred in preds.items():
                for j in range(s):
                    filas.append(
                        {**base, "serie_id": series[j], "modelo": nombre, "y_real": y_real[j], "y_pred": float(y_pred[j])}
                    )
        print(f"  horizonte {h} listo ({(time.time() - inicio) / 60:.1f} min)", flush=True)
    largo = pd.DataFrame(filas)
    largo["error"] = largo["y_pred"] - largo["y_real"]
    largo["error_abs"] = largo["error"].abs()
    return largo


def validar(panel: pd.DataFrame, largo: pd.DataFrame) -> dict:
    """Comprobaciones de integridad. Lanza AssertionError si algo falla."""
    t, s = panel.shape
    resumen = {"series": s, "meses": t}
    modelos = ["ingenuo", *CONFIGURACIONES]
    for h in HORIZONTES:
        esperados = (t - VENTANA_INICIAL - h + 1) * s
        for nombre in modelos:
            obtenidos = ((largo["horizonte"] == h) & (largo["modelo"] == nombre)).sum()
            assert obtenidos == esperados, f"{nombre} h={h}: {obtenidos} filas, se esperaban {esperados}"
        resumen[f"origenes_h{h}"] = t - VENTANA_INICIAL - h + 1
    assert not largo[["y_real", "y_pred", "error_abs"]].isna().any().any(), "hay valores nulos"
    assert np.isfinite(largo["y_pred"]).all(), "hay pronósticos no finitos"
    assert (largo["fecha_max_objetivo_train"] <= largo["origen"]).all(), "el entrenamiento usó objetivos posteriores al origen"
    assert (largo["fecha_objetivo"] > largo["origen"]).all(), "objetivo no posterior al origen"
    meses = (
        (largo["fecha_objetivo"].dt.year - largo["origen"].dt.year) * 12
        + (largo["fecha_objetivo"].dt.month - largo["origen"].dt.month)
    )
    assert (meses == largo["horizonte"]).all(), "salto de meses distinto del horizonte"
    ing = largo[largo["modelo"] == "ingenuo"]
    esperado = panel.stack().reindex(pd.MultiIndex.from_arrays([ing["origen"], ing["serie_id"]])).to_numpy()
    assert np.allclose(ing["y_pred"].to_numpy(), esperado), "el ingenuo no usa el valor del origen"
    for origen in largo["origen"].drop_duplicates():
        esperado_reg = regimen(panel[SERIE_NACIONAL].loc[:origen]).iloc[-1]
        obtenido_reg = largo.loc[largo["origen"] == origen, "regimen_origen"].iloc[0]
        assert esperado_reg == obtenido_reg, f"régimen distinto en el origen {origen:%Y-%m}"
    return resumen


def resumen_series(largo: pd.DataFrame) -> pd.DataFrame:
    """MAE por serie, horizonte y modelo, y MAE relativo al ingenuo."""
    g = largo.groupby(["serie_id", "horizonte", "modelo"], observed=True)["error_abs"].agg(["size", "mean"])
    g = g.rename(columns={"size": "n", "mean": "mae"}).reset_index()
    ref = g[g["modelo"] == "ingenuo"].set_index(["serie_id", "horizonte"])["mae"].rename("mae_ingenuo")
    g = g.join(ref, on=["serie_id", "horizonte"])
    g["mae_relativo"] = g["mae"] / g["mae_ingenuo"]
    return g[g["modelo"] != "ingenuo"].sort_values(["modelo", "horizonte", "serie_id"]).reset_index(drop=True)


def resumen_panel(largo: pd.DataFrame) -> pd.DataFrame:
    """Resumen del panel por modelo, horizonte y régimen nacional del origen (más 'total')."""
    l = largo.copy()
    l["regimen_origen"] = l["regimen_origen"].astype(str)
    todo = pd.concat([l, l.assign(regimen_origen="total")], ignore_index=True)
    por_serie = todo.groupby(["regimen_origen", "horizonte", "modelo", "serie_id"], observed=True)["error_abs"].mean().rename("mae").reset_index()
    ref = por_serie[por_serie["modelo"] == "ingenuo"].set_index(["regimen_origen", "horizonte", "serie_id"])["mae"].rename("mae_ingenuo")
    por_serie = por_serie.join(ref, on=["regimen_origen", "horizonte", "serie_id"])
    por_serie = por_serie[por_serie["modelo"] != "ingenuo"]
    por_serie["relativo"] = por_serie["mae"] / por_serie["mae_ingenuo"]
    filas = []
    for (reg, h, modelo), g in por_serie.groupby(["regimen_origen", "horizonte", "modelo"]):
        filas.append(
            {
                "modelo": modelo, "horizonte": h, "regimen_origen": reg, "series": len(g),
                "razon_error_agregada": g["mae"].sum() / g["mae_ingenuo"].sum(),
                "mediana_mae_relativo": g["relativo"].median(),
                "media_mae_relativo": g["relativo"].mean(),
                "pct_series_mejor_que_ingenuo": 100 * (g["relativo"] < 1).mean(),
            }
        )
    orden = {"baja": 0, "media": 1, "alta": 2, "total": 3}
    res = pd.DataFrame(filas)
    res["_o"] = res["regimen_origen"].map(orden)
    return res.sort_values(["horizonte", "modelo", "_o"]).drop(columns="_o").reset_index(drop=True)


def main(entrada: Path = ENTRADA_POR_DEFECTO) -> None:
    entrada = Path(entrada)
    if not entrada.exists():
        raise FileNotFoundError(f"No se encontró {entrada}. Ejecuta antes: python src/ingest.py")
    panel = cargar_panel(entrada)
    inicio = time.time()
    largo = backtest_global(panel)
    resumen = validar(panel, largo)
    minutos = (time.time() - inicio) / 60

    nac = largo[largo["serie_id"] == SERIE_NACIONAL].copy()
    nac["regimen_origen"] = nac["regimen_origen"].astype(str)
    columnas = ["origen", "horizonte", "fecha_objetivo", "modelo", "y_real", "y_pred", "error",
                "error_abs", "regimen_origen", "n_train", "fecha_max_objetivo_train"]
    nac_gb = nac[nac["modelo"] != "ingenuo"][columnas]
    met_nac = metricas(nac)
    met_series = resumen_series(largo)
    met_panel = resumen_panel(largo)

    SALIDA_PRON_NAC.parent.mkdir(parents=True, exist_ok=True)
    nac_gb.to_csv(SALIDA_PRON_NAC, index=False, encoding="utf-8")
    met_nac.to_csv(SALIDA_MET_NAC, index=False, encoding="utf-8")
    met_series.to_csv(SALIDA_MET_SERIES, index=False, encoding="utf-8")
    met_panel.to_csv(SALIDA_MET_PANEL, index=False, encoding="utf-8")

    lineas: list[str] = []

    def emitir(texto: str = "") -> None:
        print(texto)
        lineas.append(texto)

    emitir(f"Panel: {resumen['series']} series, {resumen['meses']} meses")
    emitir(f"  origenes_h1: {resumen['origenes_h1']}")
    emitir(f"  origenes_h3: {resumen['origenes_h3']}")
    emitir(f"  origenes_h6: {resumen['origenes_h6']}")
    emitir(f"Configuraciones: {', '.join(CONFIGURACIONES)}")
    emitir(f"Pronósticos nacionales: {len(nac_gb):,} filas -> {SALIDA_PRON_NAC.name}")
    emitir(f"Métricas nacionales: {len(met_nac)} filas -> {SALIDA_MET_NAC.name}")
    emitir(f"Métricas por serie: {len(met_series)} filas -> {SALIDA_MET_SERIES.name}")
    emitir(f"Métricas del panel: {len(met_panel)} filas -> {SALIDA_MET_PANEL.name}")
    emitir()
    vista = met_nac[["horizonte", "modelo", "regimen_origen", "n", "mae", "mae_relativo"]].copy()
    vista[["mae", "mae_relativo"]] = vista[["mae", "mae_relativo"]].round(3)
    emitir("Nivel general nacional: MAE (pp) y MAE relativo al ingenuo")
    emitir(vista.to_string(index=False))
    emitir()
    vp = met_panel[["horizonte", "modelo", "regimen_origen", "series", "razon_error_agregada",
                    "mediana_mae_relativo", "pct_series_mejor_que_ingenuo"]].copy()
    vp[["razon_error_agregada", "mediana_mae_relativo", "pct_series_mejor_que_ingenuo"]] = vp[
        ["razon_error_agregada", "mediana_mae_relativo", "pct_series_mejor_que_ingenuo"]].round(3)
    emitir("Panel de 126 series (régimen nacional del origen)")
    emitir(vp.to_string(index=False))
    emitir(f"\nTiempo de ajuste: {minutos:.1f} min")
    SALIDA_REGISTRO.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--entrada", type=Path, default=ENTRADA_POR_DEFECTO)
    args = parser.parse_args()
    main(args.entrada)
