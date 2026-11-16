-- Esquema del proyecto de pronóstico de inflación (IPC INDEC). Se recrea completo en cada carga.
DROP SCHEMA IF EXISTS inflacion CASCADE;
CREATE SCHEMA inflacion;

-- Dimensión de modelos (fase de origen y si es modelo principal fijado de antemano)
CREATE TABLE inflacion.modelos (
    modelo      text PRIMARY KEY,
    fase_origen text NOT NULL,
    tipo        text NOT NULL,
    principal   boolean NOT NULL
);

-- Serie del IPC en formato largo (126 series = 18 códigos x 7 regiones)
CREATE TABLE inflacion.serie_ipc (
    fecha          date NOT NULL,
    serie_id       text NOT NULL,
    codigo         text NOT NULL,
    descripcion    text,
    clasificador   text,
    region         text NOT NULL,
    indice         numeric,
    var_mensual    numeric,
    var_interanual numeric,
    PRIMARY KEY (fecha, serie_id)
);

-- Pronósticos del nivel general nacional (Fases 2, 3A y 3B), un origen por fila y modelo
CREATE TABLE inflacion.pronosticos_nacional (
    origen         date NOT NULL,
    horizonte      smallint NOT NULL CHECK (horizonte IN (1, 3, 6)),
    fecha_objetivo date NOT NULL,
    modelo         text NOT NULL REFERENCES inflacion.modelos (modelo),
    y_real         numeric NOT NULL,
    y_pred         numeric NOT NULL,
    error          numeric NOT NULL,
    error_abs      numeric NOT NULL,
    regimen_origen text NOT NULL CHECK (regimen_origen IN ('baja', 'media', 'alta')),
    PRIMARY KEY (origen, horizonte, modelo)
);

-- MAE por modelo, horizonte y régimen (nivel general nacional)
CREATE TABLE inflacion.metricas_nacional (
    modelo            text NOT NULL REFERENCES inflacion.modelos (modelo),
    horizonte         smallint NOT NULL,
    regimen_origen    text NOT NULL CHECK (regimen_origen IN ('baja', 'media', 'alta', 'total')),
    n                 integer NOT NULL,
    mae               numeric NOT NULL,
    mediana_error_abs numeric,
    p90_error_abs     numeric,
    desvio_error_abs  numeric,
    sesgo             numeric,
    mae_ingenuo       numeric,
    mae_relativo      numeric,
    PRIMARY KEY (modelo, horizonte, regimen_origen)
);

-- MAE relativo al ingenuo por serie del panel (modelos de boosting)
CREATE TABLE inflacion.metricas_series (
    serie_id     text NOT NULL,
    horizonte    smallint NOT NULL,
    modelo       text NOT NULL REFERENCES inflacion.modelos (modelo),
    n            integer NOT NULL,
    mae          numeric NOT NULL,
    mae_ingenuo  numeric NOT NULL,
    mae_relativo numeric NOT NULL,
    PRIMARY KEY (serie_id, horizonte, modelo)
);

-- Fase 4: comparación contra el ingenuo con incertidumbre
CREATE TABLE inflacion.evaluacion_resumen (
    modelo             text NOT NULL REFERENCES inflacion.modelos (modelo),
    horizonte          smallint NOT NULL,
    regimen_origen     text NOT NULL,
    n                  integer NOT NULL,
    mae                numeric NOT NULL,
    mae_ingenuo        numeric NOT NULL,
    dif_mae            numeric NOT NULL,
    ic95_inf           numeric,
    ic95_sup           numeric,
    ic_excluye_cero    boolean NOT NULL,
    pct_origenes_mejor numeric NOT NULL,
    principal          boolean NOT NULL,
    PRIMARY KEY (modelo, horizonte, regimen_origen)
);

CREATE TABLE inflacion.evaluacion_dm_total (
    modelo    text NOT NULL REFERENCES inflacion.modelos (modelo),
    horizonte smallint NOT NULL,
    n         integer NOT NULL,
    rezago    integer NOT NULL,
    dm        numeric,
    p_valor   numeric,
    principal boolean NOT NULL,
    PRIMARY KEY (modelo, horizonte)
);

CREATE TABLE inflacion.evaluacion_sensibilidad (
    modelo      text NOT NULL REFERENCES inflacion.modelos (modelo),
    horizonte   smallint NOT NULL,
    muestra     text NOT NULL CHECK (muestra IN ('completo', 'sin_salto_2023')),
    n           integer NOT NULL,
    mae         numeric NOT NULL,
    mae_ingenuo numeric NOT NULL,
    dif_mae     numeric NOT NULL,
    ic95_inf    numeric,
    ic95_sup    numeric,
    principal   boolean NOT NULL,
    PRIMARY KEY (modelo, horizonte, muestra)
);

CREATE TABLE inflacion.evaluacion_cuantiles (
    modelo         text NOT NULL REFERENCES inflacion.modelos (modelo),
    horizonte      smallint NOT NULL,
    regimen_origen text NOT NULL,
    n              integer NOT NULL,
    q50            numeric NOT NULL,
    q80            numeric NOT NULL,
    q90            numeric NOT NULL,
    PRIMARY KEY (modelo, horizonte, regimen_origen)
);

CREATE TABLE inflacion.evaluacion_panel_distribucion (
    modelo           text NOT NULL REFERENCES inflacion.modelos (modelo),
    horizonte        smallint NOT NULL,
    series           integer NOT NULL,
    p10              numeric,
    mediana          numeric,
    p90              numeric,
    pct_series_mejor numeric,
    razon_agregada   numeric,
    PRIMARY KEY (modelo, horizonte)
);
