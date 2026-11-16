-- Vistas para el dashboard (Power BI en modo Importar) y para las consultas de análisis.

-- Evaluación contra el ingenuo con una etiqueta derivada del intervalo de 95 %:
--   mejora          el intervalo queda entero por debajo de cero (el modelo reduce el error)
--   empeora         el intervalo queda entero por encima de cero
--   indistinguible  el intervalo incluye el cero
CREATE VIEW inflacion.vw_evaluacion AS
SELECT e.modelo, m.tipo, m.fase_origen, e.principal,
       e.horizonte, e.regimen_origen, e.n,
       e.mae, e.mae_ingenuo, e.dif_mae, e.ic95_inf, e.ic95_sup,
       e.pct_origenes_mejor,
       CASE WHEN e.ic95_sup < 0 THEN 'mejora'
            WHEN e.ic95_inf > 0 THEN 'empeora'
            ELSE 'indistinguible' END AS veredicto
FROM inflacion.evaluacion_resumen e
JOIN inflacion.modelos m USING (modelo);

-- Pronósticos con el error del ingenuo al lado (mismo origen y horizonte)
CREATE VIEW inflacion.vw_pronosticos AS
SELECT p.origen, p.horizonte, p.fecha_objetivo, p.modelo, m.tipo, p.regimen_origen,
       p.y_real, p.y_pred, p.error, p.error_abs,
       i.error_abs AS error_abs_ingenuo,
       p.error_abs - i.error_abs AS dif_error_abs
FROM inflacion.pronosticos_nacional p
JOIN inflacion.modelos m USING (modelo)
JOIN inflacion.pronosticos_nacional i
  ON i.origen = p.origen AND i.horizonte = p.horizonte AND i.modelo = 'ingenuo';

-- Cuánto se equivoca el ingenuo según régimen y horizonte (para "cuánto confiar")
CREATE VIEW inflacion.vw_error_ingenuo AS
SELECT horizonte, regimen_origen, n, q50, q80, q90
FROM inflacion.evaluacion_cuantiles
WHERE modelo = 'ingenuo';
