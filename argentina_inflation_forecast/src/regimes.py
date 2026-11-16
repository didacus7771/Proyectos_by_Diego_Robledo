"""Definición fija de regímenes de inflación (decidida en la Fase 1).

Regla: el régimen en una fecha se define por el promedio de los últimos
VENTANA_MESES meses de la variación mensual, usando solo datos hasta esa fecha
(sin información futura). En la evaluación, el pronóstico se etiqueta con el
régimen vigente en su fecha de origen.

Estos umbrales se fijaron mirando la distribución de la serie completa, antes
de evaluar ningún modelo. No se modifican después de ver resultados.
"""
from __future__ import annotations

import pandas as pd

VENTANA_MESES = 3
UMBRAL_BAJA_MEDIA = 3.0  # % mensual, equivale a unos 43 % anual
UMBRAL_MEDIA_ALTA = 6.0  # % mensual, equivale a unos 101 % anual
ETIQUETAS = ("baja", "media", "alta")


def regimen(var_mensual: pd.Series) -> pd.Series:
    """Clasifica cada fecha en 'baja', 'media' o 'alta'.

    `var_mensual` debe estar ordenada por fecha. Las primeras VENTANA_MESES - 1
    fechas quedan sin etiqueta (NaN) porque no hay suficiente historia.
    """
    promedio = var_mensual.rolling(VENTANA_MESES).mean()
    etiqueta = pd.cut(
        promedio,
        bins=[-float("inf"), UMBRAL_BAJA_MEDIA, UMBRAL_MEDIA_ALTA, float("inf")],
        labels=list(ETIQUETAS),
        right=False,
    )
    return etiqueta.astype(
        pd.CategoricalDtype(categories=list(ETIQUETAS), ordered=True)
    )
