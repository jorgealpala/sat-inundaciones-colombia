"""
============================================================================
 23_prediccion_vivo_imerg.py
 Proyecto: SAT de inundaciones — fuente IMERG (TIEMPO REAL)
 ----------------------------------------------------------------------------
 MODO EN VIVO con IMERG: trae precipitación IMERG reciente desde GEE
 (latencia de horas, no semanas como CHIRPS), construye features, aplica los
 3 modelos IMERG y genera las predicciones de alerta.

 IMERG V07 da precipitación en mm/hr cada 30 min. La conversión a mm/día se
 hace EN GEE: suma de (precip * 0.5) sobre las imágenes del día.

 IDEAM: nulo en vivo (no disponible); XGBoost lo tolera.

 USO:
   python 23_prediccion_vivo_imerg.py              -> última fecha disponible
   python 23_prediccion_vivo_imerg.py 2026-02-15   -> fecha manual (validación)

 Requiere: earthengine-api, pandas, numpy, joblib + gee-key.json
 Salida:   dashboard/predicciones_vivo_imerg.parquet
============================================================================
"""

import ee
import json
import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
GEE_PROJECT = os.environ.get("GEE_PROJECT", "geovolcanes-scr-piloto")
RUTA_KEY    = Path(os.environ.get("GEE_KEY_PATH", "gee-key.json"))
ASSET_MUNICIPIOS = "projects/ee-jorgealpala/assets/municipios_simplificado"
CAMPO_CODIGO = "MpCodigo"

# Rutas configurables por entorno (funcionan en local y en GitHub Actions)
DIR_OUT      = Path(os.environ.get("SAT_DIR_OUT", r"..\outputs"))
DIR_MODELOS  = Path(os.environ.get("SAT_DIR_MODELOS", str(DIR_OUT / "modelos_imerg")))
DIR_DASH     = Path(os.environ.get("SAT_DIR_DASH", str(DIR_OUT / "dashboard")))
RUTA_GEOFEAT = Path(os.environ.get("SAT_RUTA_GEOFEAT", str(DIR_OUT / "geografia_municipios.csv")))

PCT_ALTA, PCT_MEDIA = 0.95, 0.80
PISO_PERCENTIL_GLOBAL = 0.70
HORIZONTES = ["alerta_24h", "alerta_48h", "alerta_72h"]
ESCALA_IMERG = 11132                              # resolución nativa IMERG
DATASET_IMERG = "NASA/GPM_L3/IMERG_V07"


def log(msg):
    print(f"[vivo-imerg] {msg}")


# ---------------------------------------------------------------------------
# 1. AUTENTICACIÓN GEE
# ---------------------------------------------------------------------------
def inicializar_gee():
    key_json = os.environ.get("GEE_KEY_JSON")
    if key_json:
        info = json.loads(key_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=key_json)
        log("Autenticando con GEE_KEY_JSON (variable de entorno)...")
    elif RUTA_KEY.exists():
        with open(RUTA_KEY) as f:
            info = json.load(f)
        creds = ee.ServiceAccountCredentials(info["client_email"], str(RUTA_KEY))
        log(f"Autenticando con archivo {RUTA_KEY}...")
    else:
        raise FileNotFoundError("No se encontró credencial GEE.")
    ee.Initialize(creds, project=GEE_PROJECT)
    log("GEE inicializado correctamente.")


def ultima_fecha_imerg(min_imagenes=40):
    """
    Última fecha con un día COMPLETO en IMERG V07.

    IMERG genera 48 imágenes de 30 min por día. El día en curso (y a veces el
    anterior) está incompleto. Para un acumulado diario fiable, se retrocede
    hasta encontrar un día con al menos `min_imagenes` imágenes (~día completo).
    """
    col = ee.ImageCollection(DATASET_IMERG)
    ultima = ee.Date(col.aggregate_max("system:time_start"))
    fecha_str = ultima.format("YYYY-MM-dd").getInfo()
    cand = datetime.strptime(fecha_str, "%Y-%m-%d").date()

    # Retroceder hasta hallar un día con suficientes imágenes (máx 5 intentos)
    for _ in range(5):
        ini = ee.Date(cand.strftime("%Y-%m-%d"))
        fin = ini.advance(1, "day")
        n = col.filterDate(ini, fin).size().getInfo()
        if n >= min_imagenes:
            return cand
        log(f"  {cand} incompleto ({n}/48 imágenes), retrocediendo un día...")
        cand = cand - pd.Timedelta(days=1)
        cand = cand.date() if hasattr(cand, "date") else cand
    return cand


# ---------------------------------------------------------------------------
# 2. EXTRAER FEATURES IMERG POR MUNICIPIO (acumulados en mm/día, en GEE)
# ---------------------------------------------------------------------------
def extraer_features_imerg(fecha_objetivo):
    municipios = ee.FeatureCollection(ASSET_MUNICIPIOS)
    fin = ee.Date(fecha_objetivo.strftime("%Y-%m-%d"))   # día objetivo T

    imerg = ee.ImageCollection(DATASET_IMERG).select("precipitation")

    def mm_dia(f0):
        """Acumulado mm/día = suma de (mm/hr * 0.5h) en las 48 imágenes del día."""
        f1 = f0.advance(1, "day")
        return imerg.filterDate(f0, f1).map(lambda im: im.multiply(0.5)).sum()

    def img_dia(dias_atras):
        # dias_atras puede ser int (Python) o ee.Number; normalizar a ee.Number
        n = ee.Number(dias_atras)
        f0 = fin.advance(n.multiply(-1), "day")
        return mm_dia(f0)

    def img_rango(dias_ventana):
        """Suma de mm/día sobre los últimos 'dias_ventana' días (incluye T)."""
        ini = fin.advance(-(dias_ventana - 1), "day")
        f = fin.advance(1, "day")
        # mm acumulado del rango = suma de todas las medias horas * 0.5
        return imerg.filterDate(ini, f).map(lambda im: im.multiply(0.5)).sum()

    t0   = img_dia(0).rename("precip_t0")
    lag1 = img_dia(1).rename("precip_lag1")
    lag2 = img_dia(2).rename("precip_lag2")
    lag3 = img_dia(3).rename("precip_lag3")
    a3   = img_rango(3).rename("precip_acum_3d")
    a7   = img_rango(7).rename("precip_acum_7d")
    a15  = img_rango(15).rename("precip_acum_15d")
    a30  = img_rango(30).rename("precip_acum_30d")

    # máximo diario en ventana de 7 días: construir colección de 7 imágenes mm/día
    dias7 = ee.List.sequence(0, 6)
    col7 = ee.ImageCollection.fromImages(dias7.map(
        lambda d: img_dia(ee.Number(d))))
    max7 = col7.max().rename("precip_max_7d")

    stack = t0.addBands([lag1, lag2, lag3, a3, a7, a15, a30, max7])

    log("Reduciendo features IMERG por municipio en GEE...")
    fc = stack.reduceRegions(collection=municipios,
                             reducer=ee.Reducer.mean(),
                             scale=ESCALA_IMERG)

    datos = fc.getInfo()["features"]
    filas = []
    for d in datos:
        p = d["properties"]
        filas.append({
            "cod_dane": str(p.get(CAMPO_CODIGO)).zfill(5),
            "precip_t0": p.get("precip_t0"),
            "precip_lag1": p.get("precip_lag1"),
            "precip_lag2": p.get("precip_lag2"),
            "precip_lag3": p.get("precip_lag3"),
            "precip_acum_3d": p.get("precip_acum_3d"),
            "precip_acum_7d": p.get("precip_acum_7d"),
            "precip_acum_15d": p.get("precip_acum_15d"),
            "precip_acum_30d": p.get("precip_acum_30d"),
            "precip_max_7d": p.get("precip_max_7d"),
        })
    df = pd.DataFrame(filas)
    # Excluir islas (consistencia con el entrenamiento: 1120 municipios)
    df = df[~df["cod_dane"].isin(["88001", "88564"])]
    cols_num = ["precip_t0", "precip_lag1", "precip_lag2", "precip_lag3",
                "precip_acum_3d", "precip_acum_7d", "precip_acum_15d",
                "precip_acum_30d", "precip_max_7d"]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols_num)
    log(f"  Features IMERG para {len(df)} municipios.")
    return df


# ---------------------------------------------------------------------------
# 3. COMPLETAR FEATURES (IDEAM nulo + geografía)
# ---------------------------------------------------------------------------
def construir_features(imerg_feat, fecha_objetivo, geo_feat):
    df = imerg_feat.copy()
    df["precip_chirps"] = df["precip_t0"]      # nombre genérico (contiene IMERG)
    UMBRAL = 20.0
    df["dias_desde_lluvia_fuerte"] = np.where(df["precip_max_7d"] >= UMBRAL, 3, 999)
    df["mes"] = fecha_objetivo.month
    df["dia_anio"] = fecha_objetivo.timetuple().tm_yday
    for c in ["precip_ideam_t0", "precip_ideam_lag1", "precip_ideam_lag2",
              "precip_ideam_lag3", "precip_ideam_acum_3d", "precip_ideam_acum_7d",
              "precip_ideam_acum_15d", "precip_ideam_acum_30d", "dif_chirps_ideam"]:
        df[c] = np.nan
    df["tiene_ideam"] = 0
    df = df.merge(geo_feat, on="cod_dane", how="left")
    return df


# ---------------------------------------------------------------------------
# 4. NIVELES
# ---------------------------------------------------------------------------
def asignar_niveles(serie_prob, piso):
    p95 = serie_prob.quantile(PCT_ALTA)
    p80 = serie_prob.quantile(PCT_MEDIA)
    niveles = pd.Series("baja", index=serie_prob.index)
    niveles[(serie_prob >= p80) & (serie_prob >= piso)] = "media"
    niveles[(serie_prob >= p95) & (serie_prob >= piso)] = "alta"
    return niveles


def predecir_fecha(fecha_objetivo, geo_feat, modelos):
    """Genera predicciones para una sola fecha. Devuelve DataFrame o None."""
    imerg_feat = extraer_features_imerg(fecha_objetivo)
    if len(imerg_feat) == 0:
        return None
    feats_df = construir_features(imerg_feat, fecha_objetivo, geo_feat)
    salida = feats_df[["cod_dane"]].copy()
    salida["fecha"] = pd.Timestamp(fecha_objetivo)
    salida["hubo_inundacion"] = 0
    salida["precip_chirps"] = feats_df["precip_chirps"].round(2).values
    for h in HORIZONTES:
        pk = modelos[h]
        proba = pk["modelo"].predict_proba(feats_df[pk["features"]])[:, 1]
        suf = h.replace("alerta_", "")
        salida[f"prob_{suf}"] = proba.round(4)
        piso = float(pd.Series(proba).quantile(PISO_PERCENTIL_GLOBAL))
        salida[f"nivel_{suf}"] = asignar_niveles(pd.Series(proba), piso).values
    return salida


def cargar_geo_y_modelos():
    """Carga la geografía estática y los 3 modelos IMERG una sola vez."""
    geo_feat = pd.read_csv(RUTA_GEOFEAT, dtype={"cod_dane": str})
    geo_feat["cod_dane"] = geo_feat["cod_dane"].str.zfill(5)
    if "pendiente_min" in geo_feat.columns:
        geo_feat = geo_feat.drop(columns=["pendiente_min"])
    for c in ["elev_media", "elev_min"]:
        geo_feat[c] = geo_feat[c].clip(lower=0)
    modelos = {h: joblib.load(DIR_MODELOS / f"modelo_imerg_{h}.pkl")
               for h in HORIZONTES}
    return geo_feat, modelos


def main_rango(n_dias=15, fechas_extra=None, salida_nombre="predicciones_vivo_imerg.parquet"):
    """
    Genera predicciones para los últimos n_dias disponibles en IMERG
    (más fechas_extra opcionales, ej. el frente frío), y las guarda en
    un único parquet navegable por el dashboard.
    """
    DIR_DASH.mkdir(parents=True, exist_ok=True)
    inicializar_gee()
    geo_feat, modelos = cargar_geo_y_modelos()

    ultima = ultima_fecha_imerg()
    log(f"Última fecha IMERG: {ultima}")
    fechas = [ultima - pd.Timedelta(days=i) for i in range(n_dias)]
    fechas = [f.date() if hasattr(f, "date") else f for f in fechas]
    if fechas_extra:
        fechas = list(fechas_extra) + fechas
    fechas = sorted(set(fechas))

    partes = []
    for f in fechas:
        log(f"Prediciendo {f}...")
        try:
            r = predecir_fecha(f, geo_feat, modelos)
            if r is not None:
                partes.append(r)
            else:
                log(f"  Sin datos para {f}, se omite.")
        except Exception as e:
            log(f"  Error en {f}: {e}")

    if not partes:
        log("No se generaron predicciones.")
        return
    salida = pd.concat(partes, ignore_index=True)
    salida["fecha"] = salida["fecha"].dt.date
    ruta = DIR_DASH / salida_nombre
    salida.to_parquet(ruta, index=False)
    log(f"GUARDADO: {ruta}  ({salida['fecha'].nunique()} fechas, "
        f"{len(salida)} filas)")
    print("\n  Fechas incluidas:", sorted(salida['fecha'].unique()))


def main_actualizar(ventana_dias=30, salida_nombre="predicciones_vivo_imerg.parquet"):
    """
    Modo ACTUALIZACIÓN AUTOMÁTICA (para GitHub Actions, 2x día):
      - Mantiene una ventana móvil de los últimos `ventana_dias` días.
      - Conserva SIEMPRE las fechas del frente frío feb-2026 (caso fijo de
        validación), aunque queden fuera de la ventana.
      - Solo consulta a GEE las fechas que faltan en el parquet existente
        (eficiente: no recalcula lo ya hecho).
    """
    DIR_DASH.mkdir(parents=True, exist_ok=True)
    inicializar_gee()
    geo_feat, modelos = cargar_geo_y_modelos()

    # Frente frío (caso fijo)
    from datetime import date
    FRENTE_FRIO = [date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 2),
                   date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 5)]

    ultima = ultima_fecha_imerg()
    log(f"Última fecha IMERG: {ultima}")

    # Fechas objetivo de la ventana móvil
    ventana = [(ultima - pd.Timedelta(days=i)) for i in range(ventana_dias)]
    ventana = [f.date() if hasattr(f, "date") else f for f in ventana]
    fechas_objetivo = sorted(set(FRENTE_FRIO + ventana))

    # Cargar lo ya calculado (para no repetir consultas a GEE)
    ruta = DIR_DASH / salida_nombre
    existente = None
    fechas_ya = set()
    if ruta.exists():
        existente = pd.read_parquet(ruta)
        existente["fecha"] = pd.to_datetime(existente["fecha"]).dt.date
        fechas_ya = set(existente["fecha"].unique())
        log(f"  Parquet existente: {len(fechas_ya)} fechas")

    # Calcular solo las fechas que faltan
    faltan = [f for f in fechas_objetivo if f not in fechas_ya]
    log(f"  Fechas a calcular (nuevas): {len(faltan)}")

    partes = []
    for f in faltan:
        log(f"Prediciendo {f}...")
        try:
            r = predecir_fecha(f, geo_feat, modelos)
            if r is not None:
                partes.append(r)
        except Exception as e:
            log(f"  Error en {f}: {e}")

    # Combinar: existente + nuevas
    todo = []
    if existente is not None:
        existente["fecha"] = pd.to_datetime(existente["fecha"])
        todo.append(existente)
    if partes:
        nuevas = pd.concat(partes, ignore_index=True)
        todo.append(nuevas)
    if not todo:
        log("No hay datos para guardar.")
        return
    salida = pd.concat(todo, ignore_index=True)
    salida["fecha"] = pd.to_datetime(salida["fecha"]).dt.date

    # Quitar duplicados y aplicar ventana (conservando frente frío)
    salida = salida.drop_duplicates(subset=["cod_dane", "fecha"])
    corte = ultima - pd.Timedelta(days=ventana_dias)
    corte = corte.date() if hasattr(corte, "date") else corte
    mantener = (salida["fecha"] >= corte) | (salida["fecha"].isin(FRENTE_FRIO))
    salida = salida[mantener].reset_index(drop=True)

    salida.to_parquet(ruta, index=False)
    log(f"GUARDADO: {ruta}  ({salida['fecha'].nunique()} fechas, {len(salida)} filas)")
    print("\n  Fechas:", sorted(salida["fecha"].unique()))


def main(fecha_objetivo=None):
    DIR_DASH.mkdir(parents=True, exist_ok=True)
    inicializar_gee()

    if fecha_objetivo is None:
        fecha_objetivo = ultima_fecha_imerg()
        log(f"Última fecha IMERG disponible: {fecha_objetivo}")
    log(f"Fecha objetivo: {fecha_objetivo}")

    geo_feat = pd.read_csv(RUTA_GEOFEAT, dtype={"cod_dane": str})
    geo_feat["cod_dane"] = geo_feat["cod_dane"].str.zfill(5)
    if "pendiente_min" in geo_feat.columns:
        geo_feat = geo_feat.drop(columns=["pendiente_min"])
    for c in ["elev_media", "elev_min"]:
        geo_feat[c] = geo_feat[c].clip(lower=0)

    imerg_feat = extraer_features_imerg(fecha_objetivo)
    if len(imerg_feat) == 0:
        log("ERROR: IMERG no devolvió datos para esa fecha. Prueba otra fecha.")
        return
    feats_df = construir_features(imerg_feat, fecha_objetivo, geo_feat)
    log(f"Features construidas para {len(feats_df)} municipios.")

    salida = feats_df[["cod_dane"]].copy()
    salida["fecha"] = pd.Timestamp(fecha_objetivo)
    salida["hubo_inundacion"] = 0
    salida["precip_chirps"] = feats_df["precip_chirps"].round(2).values

    for h in HORIZONTES:
        pk = joblib.load(DIR_MODELOS / f"modelo_imerg_{h}.pkl")
        modelo, model_feats = pk["modelo"], pk["features"]
        proba = modelo.predict_proba(feats_df[model_feats])[:, 1]
        suf = h.replace("alerta_", "")
        salida[f"prob_{suf}"] = proba.round(4)
        piso = float(pd.Series(proba).quantile(PISO_PERCENTIL_GLOBAL))
        salida[f"nivel_{suf}"] = asignar_niveles(pd.Series(proba), piso).values

    ruta = DIR_DASH / "predicciones_vivo_imerg.parquet"
    salida["fecha"] = salida["fecha"].dt.date
    salida.to_parquet(ruta, index=False)
    log(f"GUARDADO: {ruta}  ({len(salida)} municipios)")
    print("\n  Distribución niveles 24h:")
    print(salida["nivel_24h"].value_counts().to_string())


if __name__ == "__main__":
    from datetime import date
    # Modos de uso:
    #   python 23_prediccion_vivo_imerg.py              -> últimos 15 días (navegable)
    #   python 23_prediccion_vivo_imerg.py 2026-02-15   -> una fecha específica
    #   python 23_prediccion_vivo_imerg.py frentefrio   -> 15 días + frente frío feb 2026
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "frentefrio":
            ff = [date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 2),
                  date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 5)]
            main_rango(n_dias=15, fechas_extra=ff)
        elif arg == "actualizar":
            # Modo para GitHub Actions: ventana móvil 30d + frente frío
            main_actualizar(ventana_dias=30)
        else:
            main(datetime.strptime(arg, "%Y-%m-%d").date())
    else:
        main_rango(n_dias=15)
