"""Fase 4: evaluación por régimen y horizonte, con incertidumbre.

Lee los pronósticos nacionales ya generados (Fases 2, 3A y 3B) y NO vuelve a ajustar ningún
modelo. Para cada modelo, horizonte y régimen compara contra el ingenuo con la diferencia de
error absoluto por origen:  d = |error del modelo| - |error del ingenuo|  (d < 0: el modelo
mejora al ingenuo). Escribe en data/processed/:
    evaluacion_resumen.csv           MAE, diferencia vs ingenuo e intervalo de 95 %, por régimen
    evaluacion_dm_total.csv          prueba de Diebold-Mariano (total, varianza de largo plazo)
    evaluacion_sensibilidad.csv      total con y sin los orígenes del salto de 2023
    evaluacion_cuantiles.csv         cuantiles 50/80/90 del error absoluto (cuánto confiar)
    evaluacion_panel_distribucion.csv  distribución del MAE relativo entre las 126 series
    evaluacion_salida.txt            copia de lo que imprime

Definiciones fijas (se declaran antes de ver resultados y no se cambian):
    - Intervalo: bootstrap por bloques móviles de d, 5000 remuestreos, semilla 0, percentiles
      2,5 y 97,5. Longitud de bloque: 3 meses a h=1 y h=3, 6 meses a h=6. Dentro de un régimen
      los orígenes se ordenan por fecha y se tratan como secuencia (aproximación: la alta
      tiene solo dos episodios y n pequeño, el intervalo es indicativo).
    - Diebold-Mariano: pérdida absoluta, varianza de largo plazo con núcleo de Bartlett y
      rezago h-1, corrección de Harvey-Leybourne-Newbold, p-valor con t de n-1 gr. de libertad
      (bilateral).
    - Salto de 2023 (sensibilidad exploratoria): orígenes cuyo mes objetivo cae entre
      2023-08 y 2024-01 (ambos incluidos).
    - Modelos principales fijados de antemano: ingenuo (baseline) y gb_base. El resto se
      reporta como exploratorio. No se corrigen los p-valores por comparaciones múltiples.

Uso (desde la raíz del proyecto):
    python src/evaluacion.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"

ENTRADAS = [
    PROC / "backtest_baselines_pronosticos.csv",
    PROC / "backtest_modelos_pronosticos.csv",
    PROC / "backtest_global_pronosticos_nacional.csv",
]
METRICAS_REF = [
    PROC / "backtest_baselines_metricas.csv",
    PROC / "backtest_modelos_metricas.csv",
    PROC / "backtest_global_metricas_nacional.csv",
]
SERIES = PROC / "backtest_global_metricas_series.csv"

SALIDAS = {
    "resumen": PROC / "evaluacion_resumen.csv",
    "dm": PROC / "evaluacion_dm_total.csv",
    "sens": PROC / "evaluacion_sensibilidad.csv",
    "cuant": PROC / "evaluacion_cuantiles.csv",
    "panel": PROC / "evaluacion_panel_distribucion.csv",
    "txt": PROC / "evaluacion_salida.txt",
}

BASELINE = "ingenuo"
PRINCIPALES = ("ingenuo", "gb_base")
HORIZONTES = (1, 3, 6)
REGIMENES = ("baja", "media", "alta", "total")
ORIGENES_ESPERADOS = {1: 81, 3: 79, 6: 76}
BLOQUE = {1: 3, 3: 3, 6: 6}
REMUESTREOS = 5000
SEMILLA = 0
SHOCK_DESDE, SHOCK_HASTA = pd.Timestamp("2023-08-01"), pd.Timestamp("2024-01-01")
COLUMNAS_COMUNES = ["origen", "horizonte", "fecha_objetivo", "modelo", "y_real", "y_pred",
                    "error", "error_abs", "regimen_origen"]

_lineas: list[str] = []


def emitir(texto: str = "") -> None:
    print(texto)
    _lineas.append(texto)


def cargar() -> pd.DataFrame:
    partes = [pd.read_csv(r, usecols=COLUMNAS_COMUNES, parse_dates=["origen", "fecha_objetivo"])
              for r in ENTRADAS]
    base = partes[0]
    # El ingenuo viene de la Fase 2. Si otra fuente también lo trae, se comprueba que sea
    # idéntico y se conserva una sola copia.
    ref = base[base.modelo == BASELINE].sort_values(["horizonte", "origen"]).reset_index(drop=True)
    for p in partes[1:]:
        otro = p[p.modelo == BASELINE].sort_values(["horizonte", "origen"]).reset_index(drop=True)
        if len(otro) == 0:
            continue
        if len(otro) != len(ref) or np.abs(otro.error_abs.to_numpy() - ref.error_abs.to_numpy()).max() > 1e-9:
            raise SystemExit("El ingenuo no coincide entre las fuentes de pronósticos.")
    datos = pd.concat([partes[0]] + [p[p.modelo != BASELINE] for p in partes[1:]], ignore_index=True)
    return datos


def validar(datos: pd.DataFrame) -> None:
    if datos.duplicated(["origen", "horizonte", "modelo"]).any():
        raise SystemExit("Hay pronósticos duplicados (origen, horizonte, modelo).")
    if datos[COLUMNAS_COMUNES].isna().any().any():
        raise SystemExit("Hay valores faltantes en los pronósticos.")
    for h, n in ORIGENES_ESPERADOS.items():
        sub = datos[datos.horizonte == h]
        cuentas = sub.groupby("modelo").size()
        if (cuentas != n).any():
            raise SystemExit(f"h={h}: algún modelo no tiene {n} orígenes: {cuentas.to_dict()}")
        # el valor real debe ser el mismo para todos los modelos en cada origen
        if (sub.groupby("origen").y_real.nunique() > 1).any():
            raise SystemExit(f"h={h}: y_real distinto entre modelos en el mismo origen.")
    # el MAE recalculado debe coincidir con las métricas de cada fase (< 1e-9)
    maximo = 0.0
    for ruta in METRICAS_REF:
        ref = pd.read_csv(ruta)
        for fila in ref.itertuples():
            s = datos[(datos.modelo == fila.modelo) & (datos.horizonte == fila.horizonte)]
            if fila.regimen_origen != "total":
                s = s[s.regimen_origen == fila.regimen_origen]
            if len(s) != fila.n:
                raise SystemExit(f"n distinto para {fila.modelo} h={fila.horizonte} {fila.regimen_origen}")
            maximo = max(maximo, abs(s.error_abs.mean() - fila.mae))
    if maximo > 1e-9:
        raise SystemExit(f"El MAE recalculado difiere de las métricas previas en {maximo:.3g}.")
    emitir(f"Validación: MAE recalculado = métricas previas (diferencia máxima {maximo:.2e})")


def bootstrap_bloques(d: np.ndarray, bloque: int) -> tuple[float, float]:
    """Intervalo de 95 % de la media de d por bootstrap de bloques móviles.

    Usa un generador propio con la semilla fija en cada llamada: la misma serie d produce
    siempre el mismo intervalo, sin depender del orden de los cálculos."""
    rng = np.random.default_rng(SEMILLA)
    n = len(d)
    if n < 2:
        return (np.nan, np.nan)
    b = min(bloque, n)
    nb = int(np.ceil(n / b))
    inicios = rng.integers(0, n - b + 1, size=(REMUESTREOS, nb))
    idx = (inicios[:, :, None] + np.arange(b)[None, None, :]).reshape(REMUESTREOS, -1)[:, :n]
    medias = d[idx].mean(axis=1)
    return (float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5)))


def diebold_mariano(d: np.ndarray, h: int) -> tuple[float, float]:
    """Estadístico DM (pérdida absoluta) con Bartlett, rezago h-1 y corrección HLN."""
    n = len(d)
    dm = d - d.mean()
    lrv = float(np.dot(dm, dm) / n)
    for k in range(1, h):
        peso = 1 - k / h
        lrv += 2 * peso * float(np.dot(dm[k:], dm[:-k]) / n)
    if lrv <= 0:
        return (np.nan, np.nan)
    est = d.mean() / np.sqrt(lrv / n)
    est *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(est), df=n - 1))
    return (float(est), float(p))


def diferencias(datos: pd.DataFrame, modelo: str, h: int) -> pd.DataFrame:
    """Por origen: error absoluto del modelo, del ingenuo y d, ordenado por fecha."""
    a = datos[(datos.modelo == modelo) & (datos.horizonte == h)].set_index("origen")
    b = datos[(datos.modelo == BASELINE) & (datos.horizonte == h)].set_index("origen")
    out = pd.DataFrame({"ea_modelo": a.error_abs, "ea_ingenuo": b.error_abs,
                        "regimen": a.regimen_origen, "fecha_objetivo": a.fecha_objetivo}).sort_index()
    out["d"] = out.ea_modelo - out.ea_ingenuo
    return out


def subconjuntos(df: pd.DataFrame):
    for reg in REGIMENES:
        yield reg, (df if reg == "total" else df[df.regimen == reg])


def evaluar(datos: pd.DataFrame):
    modelos = [m for m in datos.modelo.unique() if m != BASELINE]
    resumen, dm_filas, sens, cuant = [], [], [], []
    for h in HORIZONTES:
        # cuantiles del ingenuo
        for mod in [BASELINE] + modelos:
            df = diferencias(datos, mod, h)
            for reg, sub in subconjuntos(df):
                q = sub.ea_modelo.quantile([0.5, 0.8, 0.9]).to_numpy()
                cuant.append(dict(modelo=mod, horizonte=h, regimen_origen=reg, n=len(sub),
                                  q50=q[0], q80=q[1], q90=q[2]))
        for mod in modelos:
            df = diferencias(datos, mod, h)
            for reg, sub in subconjuntos(df):
                d = sub.d.to_numpy()
                lo, hi = bootstrap_bloques(d, BLOQUE[h])
                resumen.append(dict(
                    modelo=mod, horizonte=h, regimen_origen=reg, n=len(sub),
                    mae=sub.ea_modelo.mean(), mae_ingenuo=sub.ea_ingenuo.mean(),
                    dif_mae=d.mean(), ic95_inf=lo, ic95_sup=hi,
                    ic_excluye_cero=bool(lo > 0 or hi < 0),
                    pct_origenes_mejor=100 * float((d < 0).mean()),
                    principal=mod in PRINCIPALES))
            est, p = diebold_mariano(df.d.to_numpy(), h)
            dm_filas.append(dict(modelo=mod, horizonte=h, n=len(df), rezago=h - 1,
                                 dm=est, p_valor=p, principal=mod in PRINCIPALES))
            en_shock = (df.fecha_objetivo >= SHOCK_DESDE) & (df.fecha_objetivo <= SHOCK_HASTA)
            for nombre, sub in (("completo", df), ("sin_salto_2023", df[~en_shock])):
                d = sub.d.to_numpy()
                lo, hi = bootstrap_bloques(d, BLOQUE[h])
                sens.append(dict(modelo=mod, horizonte=h, muestra=nombre, n=len(sub),
                                 mae=sub.ea_modelo.mean(), mae_ingenuo=sub.ea_ingenuo.mean(),
                                 dif_mae=d.mean(), ic95_inf=lo, ic95_sup=hi,
                                 principal=mod in PRINCIPALES))
    return (pd.DataFrame(resumen), pd.DataFrame(dm_filas), pd.DataFrame(sens), pd.DataFrame(cuant))


def distribucion_panel() -> pd.DataFrame:
    s = pd.read_csv(SERIES)
    filas = []
    for (mod, h), g in s.groupby(["modelo", "horizonte"]):
        if mod == BASELINE:
            continue
        r = g.mae_relativo
        filas.append(dict(modelo=mod, horizonte=h, series=len(g), p10=r.quantile(0.1),
                          mediana=r.median(), p90=r.quantile(0.9),
                          pct_series_mejor=100 * float((r < 1).mean()),
                          razon_agregada=g.mae.sum() / g.mae_ingenuo.sum()))
    return pd.DataFrame(filas)


def main() -> None:
    datos = cargar()
    validar(datos)
    resumen, dm, sens, cuant = evaluar(datos)
    panel = distribucion_panel()
    for clave, df in (("resumen", resumen), ("dm", dm), ("sens", sens), ("cuant", cuant), ("panel", panel)):
        df.to_csv(SALIDAS[clave], index=False, encoding="utf-8", lineterminator="\n")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    emitir(f"Filas: resumen {len(resumen)}, dm {len(dm)}, sensibilidad {len(sens)}, "
           f"cuantiles {len(cuant)}, panel {len(panel)}")
    emitir("\nDiferencia de MAE contra el ingenuo (pp; negativa = el modelo mejora), total:")
    t = resumen[resumen.regimen_origen == "total"][
        ["modelo", "horizonte", "n", "mae", "mae_ingenuo", "dif_mae", "ic95_inf", "ic95_sup"]]
    emitir(t.round(3).to_string(index=False))
    emitir("\nPor régimen, modelos principales (gb_base):")
    p = resumen[(resumen.modelo == "gb_base")][
        ["horizonte", "regimen_origen", "n", "dif_mae", "ic95_inf", "ic95_sup", "pct_origenes_mejor"]]
    emitir(p.round(3).to_string(index=False))
    emitir("\nDiebold-Mariano (total):")
    emitir(dm.round(4).to_string(index=False))
    emitir("\nSensibilidad al salto de 2023:")
    emitir(sens.round(3).to_string(index=False))
    emitir("\nCuantiles del error absoluto del ingenuo:")
    emitir(cuant[cuant.modelo == BASELINE].round(3).to_string(index=False))
    emitir("\nDistribución del MAE relativo en el panel:")
    emitir(panel.round(3).to_string(index=False))
    SALIDAS["txt"].write_text("\n".join(_lineas) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
