-- Consultas de análisis sobre el esquema inflacion. Cada una responde una pregunta del proyecto.

-- 1) ¿Algún modelo mejora al ingenuo en el total, con intervalo que excluya el cero?
SELECT modelo, horizonte, round(dif_mae, 3) AS dif_mae_pp, veredicto
FROM inflacion.vw_evaluacion
WHERE regimen_origen = 'total'
ORDER BY horizonte, dif_mae;

-- 2) ¿Qué porcentaje del error absoluto total del ingenuo aporta cada régimen? (h = 1, 3, 6)
SELECT horizonte, regimen_origen,
       count(*) AS origenes,
       round(sum(error_abs), 2) AS error_abs_total,
       round(100 * sum(error_abs) / sum(sum(error_abs)) OVER (PARTITION BY horizonte), 1) AS pct_del_error
FROM inflacion.pronosticos_nacional
WHERE modelo = 'ingenuo'
GROUP BY horizonte, regimen_origen
ORDER BY horizonte, regimen_origen;

-- 3) Concentración: los 6 orígenes con mayor error del ingenuo a h = 1 y su peso acumulado en el total
--    (son los seis peores, no una ventana fija de fechas)
WITH ing AS (
    SELECT origen, fecha_objetivo, error_abs,
           row_number() OVER (ORDER BY error_abs DESC) AS puesto,
           sum(error_abs) OVER () AS total
    FROM inflacion.pronosticos_nacional
    WHERE modelo = 'ingenuo' AND horizonte = 1
)
SELECT puesto, origen, fecha_objetivo, round(error_abs, 2) AS error_abs,
       round(100 * sum(error_abs) OVER (ORDER BY puesto) / total, 1) AS pct_acumulado
FROM ing
WHERE puesto <= 6
ORDER BY puesto;

-- 4) Panel de 126 series: ¿en qué horizontes el boosting global mejora al ingenuo en conjunto?
SELECT modelo, horizonte, round(razon_agregada, 3) AS razon_agregada,
       round(pct_series_mejor, 1) AS pct_series_mejor
FROM inflacion.evaluacion_panel_distribucion
ORDER BY horizonte, razon_agregada;
