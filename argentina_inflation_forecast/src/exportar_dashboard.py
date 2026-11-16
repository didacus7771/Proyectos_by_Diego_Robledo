"""Exporta a CSV los objetos de PostgreSQL que usa el dashboard de Power BI (Fase 5b).

Solo lectura. Lee la conexión con conexion() de cargar_postgres.py (no imprime la contraseña).
Escribe en dashboard/datos/: UTF-8 sin BOM, fin de línea LF, punto decimal, fechas ISO.
Respaldo por si Power BI Desktop no conecta a PostgreSQL; el camino principal es conectar directo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cargar_postgres import RAIZ, conexion  # noqa: E402

DESTINO = RAIZ / "dashboard" / "datos"

# objeto -> (consulta, filas esperadas)
OBJETOS = {
    "vw_evaluacion": ("SELECT * FROM inflacion.vw_evaluacion ORDER BY modelo, horizonte, regimen_origen", 108),
    "vw_pronosticos": ("SELECT * FROM inflacion.vw_pronosticos ORDER BY modelo, horizonte, origen", 2360),
    "vw_error_ingenuo": ("SELECT * FROM inflacion.vw_error_ingenuo ORDER BY horizonte, regimen_origen", 12),
    "evaluacion_panel_distribucion": (
        "SELECT * FROM inflacion.evaluacion_panel_distribucion ORDER BY modelo, horizonte", 9),
    "evaluacion_sensibilidad": (
        "SELECT * FROM inflacion.evaluacion_sensibilidad ORDER BY modelo, horizonte, muestra", 54),
    "evaluacion_dm_total": ("SELECT * FROM inflacion.evaluacion_dm_total ORDER BY modelo, horizonte", 27),
    "modelos": ("SELECT * FROM inflacion.modelos ORDER BY modelo", 10),
}


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    motor = create_engine(conexion())
    error = False
    print("Exportación para el dashboard:")
    with motor.connect() as con:
        con.execute(text("SET default_transaction_read_only = on"))
        for nombre, (sql, esperado) in OBJETOS.items():
            df = pd.read_sql(text(sql), con)
            df.to_csv(DESTINO / f"{nombre}.csv", index=False, encoding="utf-8", lineterminator="\n")
            ok = len(df) == esperado
            error |= not ok
            print(f"  {nombre:32s} filas {len(df):5d} | esperado {esperado:5d} | columnas {len(df.columns):2d} | "
                  f"{'OK' if ok else 'NO COINCIDE'}")
    print("  Resultado:", "FALLO" if error else "OK")
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
