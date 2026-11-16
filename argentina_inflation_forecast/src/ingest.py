"""Fase 1: ingesta y limpieza del IPC por divisiones (INDEC).

Lee data/raw/serie_ipc_divisiones.csv (sin modificarlo), valida su estructura
y escribe data/processed/ipc_divisiones_tidy.csv en formato ordenado.

Uso (desde la raíz del proyecto):
    python src/ingest.py
    python src/ingest.py --entrada otra_ruta.csv --salida otra_salida.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA_POR_DEFECTO = RAIZ / "data" / "raw" / "serie_ipc_divisiones.csv"
SALIDA_POR_DEFECTO = RAIZ / "data" / "processed" / "ipc_divisiones_tidy.csv"

COLUMNAS_CRUDAS = [
    "Codigo", "Descripcion", "Clasificador", "Periodo",
    "Indice_IPC", "v_m_IPC", "v_i_a_IPC", "Region",
]
REGIONES_ESPERADAS = {
    "Nacional", "GBA", "Pampeana", "Noreste", "Noroeste", "Cuyo", "Patagonia",
}
# El archivo trae la descripción vacía para estas series; se completa con el
# nombre del clasificador al que pertenecen.
DESCRIPCION_FALTANTE = {
    "B": "Bienes",
    "S": "Servicios",
    "Estacional": "Estacional",
    "Núcleo": "Núcleo",
    "Regulados": "Regulados",
}
# La variación mensual publicada tiene 1 decimal; el índice, 4. La diferencia
# máxima esperada por redondeo es de unos 0,05 puntos.
TOLERANCIA_CONSISTENCIA_PP = 0.1


def leer_crudo(ruta: Path) -> pd.DataFrame:
    """Lee el CSV del INDEC: separador ';', coma decimal, latin-1, 'NA' como nulo."""
    return pd.read_csv(
        ruta,
        sep=";",
        encoding="latin-1",
        decimal=",",
        na_values=["NA"],
        dtype={"Codigo": str},
    )


def _normalizar_texto(texto: pd.Series) -> pd.Series:
    """Quita espacios sobrantes y pasa 'NIVEL GENERAL' a 'Nivel general'."""
    limpio = texto.str.strip().str.replace(r"\s+", " ", regex=True)
    return limpio.where(~limpio.str.isupper(), limpio.str.capitalize())


def limpiar(crudo: pd.DataFrame) -> pd.DataFrame:
    """Convierte el archivo crudo en una tabla ordenada con nombres claros."""
    faltan = set(COLUMNAS_CRUDAS) - set(crudo.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el archivo: {sorted(faltan)}")

    df = crudo[COLUMNAS_CRUDAS].copy()
    df["Codigo"] = df["Codigo"].str.strip()
    df["Descripcion"] = df["Descripcion"].fillna("")
    df["Descripcion"] = _normalizar_texto(df["Descripcion"])
    vacias = df["Descripcion"] == ""
    df.loc[vacias, "Descripcion"] = df.loc[vacias, "Codigo"].map(DESCRIPCION_FALTANTE)

    periodo = df["Periodo"].astype(str)
    if not periodo.str.fullmatch(r"\d{6}").all():
        raise ValueError("Hay valores de Periodo que no tienen formato AAAAMM")
    fecha = pd.to_datetime(periodo, format="%Y%m")

    tidy = pd.DataFrame(
        {
            "fecha": fecha,
            "serie_id": df["Codigo"] + "|" + df["Region"].str.strip(),
            "codigo": df["Codigo"],
            "descripcion": df["Descripcion"],
            "clasificador": df["Clasificador"].str.strip(),
            "region": df["Region"].str.strip(),
            "indice": df["Indice_IPC"],
            "var_mensual": df["v_m_IPC"],
            "var_interanual": df["v_i_a_IPC"],
        }
    )
    return tidy.sort_values(["serie_id", "fecha"]).reset_index(drop=True)


def validar(tidy: pd.DataFrame) -> dict:
    """Comprueba la estructura de los datos. Lanza AssertionError si algo falla.

    Devuelve un resumen con las medidas de cada chequeo.
    """
    resumen: dict = {}

    assert tidy["descripcion"].notna().all() and (tidy["descripcion"] != "").all(), \
        "Hay series sin descripción"
    assert set(tidy["region"].unique()) == REGIONES_ESPERADAS, \
        f"Regiones inesperadas: {sorted(tidy['region'].unique())}"
    assert not tidy.duplicated(["serie_id", "fecha"]).any(), "Hay filas duplicadas"
    assert tidy["indice"].notna().all() and (tidy["indice"] > 0).all(), \
        "Hay índices nulos o no positivos"

    # Cada serie debe cubrir los mismos meses, sin huecos.
    meses_por_serie = tidy.groupby("serie_id")["fecha"].agg(["min", "max", "count"])
    assert meses_por_serie["min"].nunique() == 1 and meses_por_serie["max"].nunique() == 1, \
        "Las series no cubren el mismo rango de fechas"
    esperados = len(pd.period_range(meses_por_serie["min"].iloc[0],
                                    meses_por_serie["max"].iloc[0], freq="M"))
    assert (meses_por_serie["count"] == esperados).all(), "Hay meses faltantes en alguna serie"
    resumen["series"] = int(len(meses_por_serie))
    resumen["meses"] = int(esperados)
    resumen["desde"] = str(meses_por_serie["min"].iloc[0].date())
    resumen["hasta"] = str(meses_por_serie["max"].iloc[0].date())

    # Base del índice: diciembre de 2016 = 100.
    primero = tidy[tidy["fecha"] == tidy["fecha"].min()]
    assert (primero["indice"] == 100).all(), "El primer mes no tiene índice 100"

    # Los nulos solo pueden estar al inicio de cada serie.
    orden = tidy.groupby("serie_id").cumcount()
    assert tidy.loc[tidy["var_mensual"].isna(), "fecha"].eq(tidy["fecha"].min()).all(), \
        "Hay variaciones mensuales nulas fuera del primer mes"
    assert (orden[tidy["var_interanual"].isna()] < 12).all(), \
        "Hay variaciones interanuales nulas después del mes 12"

    # La variación mensual publicada debe coincidir con la calculada desde el índice.
    calculada = tidy.groupby("serie_id")["indice"].pct_change() * 100
    diferencia = (calculada - tidy["var_mensual"]).abs()
    resumen["diferencia_max_var_mensual_pp"] = round(float(diferencia.max()), 3)
    assert diferencia.max() <= TOLERANCIA_CONSISTENCIA_PP, \
        f"La variación mensual publicada difiere del índice en {diferencia.max():.3f} pp"
    return resumen


def main(entrada: Path = ENTRADA_POR_DEFECTO, salida: Path = SALIDA_POR_DEFECTO) -> pd.DataFrame:
    entrada, salida = Path(entrada), Path(salida)
    if not entrada.exists():
        raise FileNotFoundError(f"No se encontró {entrada}. Guarda el CSV original en data/raw/.")
    tidy = limpiar(leer_crudo(entrada))
    resumen = validar(tidy)
    salida.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(salida, index=False, encoding="utf-8")
    print(f"Leído: {entrada.name} -> {len(tidy):,} filas")
    for clave, valor in resumen.items():
        print(f"  {clave}: {valor}")
    print(f"Escrito: {salida}")
    return tidy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--entrada", type=Path, default=ENTRADA_POR_DEFECTO)
    parser.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    args = parser.parse_args()
    main(args.entrada, args.salida)
