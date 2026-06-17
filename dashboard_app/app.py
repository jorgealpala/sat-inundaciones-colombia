"""
============================================================================
 app.py  —  Dashboard SAT de Inundaciones (Colombia)  v4
 ----------------------------------------------------------------------------
 Cambios v4:
   - Leyenda de colores (verde/amarillo/rojo) sobre el mapa en ambas vistas.
   - Vista NACIONAL: métricas superiores solo Fecha + Inundaciones reales
     (se quitan labels alta/media que casi no cambian).
   - Vista DEPARTAMENTAL: se mantienen labels alta/media + leyenda.
   - Gráficas por departamento y de recurrencia: TOP 10.
   - Vista NACIONAL: recurrencia top 10 + estacionalidad mensual por municipio.
   - Gráficas estadísticas adicionales de interés.

 Ejecutar:  streamlit run app.py
 Estructura:
     app.py
     data/predicciones_demo.parquet        (de 14)
     data/municipios_dashboard.geojson     (de 15)
     data/recurrencia_municipios.parquet   (de 16)
     data/recurrencia_mensual.parquet      (de 16)
============================================================================
"""

import time as _time
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import altair as alt
from streamlit_folium import st_folium
from folium import Element
from pathlib import Path
from reporte_pdf import generar_reporte

st.set_page_config(page_title="SAT Inundaciones Colombia",
                   page_icon="🌧️", layout="wide")

DIR_DATA = Path(__file__).parent / "data"
RUTA_GEO  = DIR_DATA / "municipios_dashboard.geojson"
RUTA_REC  = DIR_DATA / "recurrencia_municipios.parquet"
RUTA_MENS = DIR_DATA / "recurrencia_mensual.parquet"

# Archivos de predicciones por fuente y modo
RUTAS_PRED = {
    ("IMERG", "Demo histórica"):  DIR_DATA / "predicciones_demo_imerg.parquet",
    ("IMERG", "Tiempo real"):     DIR_DATA / "predicciones_vivo_imerg.parquet",
    ("CHIRPS", "Demo histórica"): DIR_DATA / "predicciones_demo.parquet",
    ("CHIRPS", "Tiempo real"):    DIR_DATA / "predicciones_vivo.parquet",
}

# URL raw de GitHub para los datos de TIEMPO REAL (los que actualiza el bot).
# Leer desde aquí garantiza siempre la última versión, sin depender de que
# Streamlit Cloud reinicie el contenedor tras el push del workflow.
GITHUB_RAW = ("https://raw.githubusercontent.com/jorgealpala/"
              "sat-inundaciones-colombia/main/dashboard_app/data/")
URLS_PRED_VIVO = {
    "IMERG":  GITHUB_RAW + "predicciones_vivo_imerg.parquet",
    "CHIRPS": GITHUB_RAW + "predicciones_vivo.parquet",
}

COLORES = {"alta": "#d73027", "media": "#fee08b", "baja": "#1a9850"}
ORDEN_NIVEL = ["alta", "media", "baja"]
MESES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
         7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

REGIONES = {
    "Amazonas": "Amazonía", "Caquetá": "Amazonía", "Guainía": "Amazonía",
    "Guaviare": "Amazonía", "Putumayo": "Amazonía", "Vaupés": "Amazonía",
    "Atlántico": "Caribe", "Bolívar": "Caribe", "Cesar": "Caribe",
    "Córdoba": "Caribe", "La Guajira": "Caribe", "Magdalena": "Caribe",
    "Sucre": "Caribe", "San Andrés y Providencia": "Caribe",
    "Chocó": "Pacífico",
    "Arauca": "Orinoquía", "Casanare": "Orinoquía", "Meta": "Orinoquía",
    "Vichada": "Orinoquía",
    "Antioquia": "Andina", "Boyacá": "Andina", "Caldas": "Andina",
    "Cauca": "Andina", "Cundinamarca": "Andina", "Huila": "Andina",
    "Nariño": "Andina", "Norte de Santander": "Andina", "Quindío": "Andina",
    "Risaralda": "Andina", "Santander": "Andina", "Tolima": "Andina",
    "Valle del Cauca": "Andina", "N/A": "Sin región",
}

LEYENDA_HTML = """
<div style="position: fixed; bottom: 30px; left: 12px; z-index: 9999;
            background: white; padding: 8px 12px; border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px;">
  <b>Nivel de alerta</b><br>
  <span style="color:#d73027;">&#9632;</span> Alta&nbsp;&nbsp;
  <span style="color:#e8c100;">&#9632;</span> Media&nbsp;&nbsp;
  <span style="color:#1a9850;">&#9632;</span> Baja<br>
  <span style="color:#ff1493;">&#9679;</span> Inundación real (UNGRD)
</div>
"""

LEYENDA_PRECIP = """
<div style="position: fixed; bottom: 30px; left: 12px; z-index: 9999;
            background: white; padding: 8px 12px; border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px;">
  <b>Precipitación (mm/día)</b><br>
  <span style="color:#f7fbff;">&#9632;</span> &lt;1&nbsp;&nbsp;
  <span style="color:#c6dbef;">&#9632;</span> 1–5&nbsp;&nbsp;
  <span style="color:#6baed6;">&#9632;</span> 5–10<br>
  <span style="color:#3182bd;">&#9632;</span> 10–20&nbsp;&nbsp;
  <span style="color:#08519c;">&#9632;</span> 20–40&nbsp;&nbsp;
  <span style="color:#08306b;">&#9632;</span> &gt;40<br>
  <span style="color:#ff1493;">&#9679;</span> Inundación real (UNGRD)
</div>
"""


@st.cache_data(ttl=900)  # expira cada 15 min como red de seguridad
def cargar_predicciones(ruta_str, _firma):
    # _firma (mtime + tamaño del archivo) fuerza recarga cuando el archivo cambia.
    # El ttl garantiza que, aun si la firma no cambiara, el caché se renueve.
    df = pd.read_parquet(ruta_str)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["cod_dane"] = df["cod_dane"].astype(str).str.zfill(5)
    return df


@st.cache_data(ttl=600)  # 10 min: lee la última versión publicada en GitHub
def cargar_predicciones_remoto(url, _ventana):
    # _ventana (bloque de 10 min) actúa como cache-buster temporal: cuando cambia,
    # se vuelve a descargar el parquet más reciente que el bot publicó en GitHub.
    # Se descarga con requests (universal) y se lee desde memoria, evitando
    # depender de fsspec en el entorno de despliegue.
    import io
    import requests
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["cod_dane"] = df["cod_dane"].astype(str).str.zfill(5)
    return df


def _firma_archivo(ruta):
    """Huella del archivo: combina fecha de modificación y tamaño en bytes.
    Si GitHub Actions reescribe el .parquet, al menos uno de los dos cambia,
    invalidando el caché aunque el sistema de archivos redondee el mtime."""
    try:
        s = ruta.stat()
        return f"{int(s.st_mtime)}_{s.st_size}"
    except Exception:
        return "0_0"

@st.cache_data
def cargar_geojson():
    g = gpd.read_file(RUTA_GEO)
    g["cod_dane"] = g["cod_dane"].astype(str).str.zfill(5)
    g["region"] = g["Depto"].map(REGIONES).fillna("Sin región")
    return g

@st.cache_data
def cargar_recurrencia():
    try:
        r = pd.read_parquet(RUTA_REC)
        r["cod_dane"] = r["cod_dane"].astype(str).str.zfill(5)
        return r
    except Exception:
        return None

@st.cache_data
def cargar_mensual():
    try:
        m = pd.read_parquet(RUTA_MENS)
        m["cod_dane"] = m["cod_dane"].astype(str).str.zfill(5)
        return m
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CABECERA
# ---------------------------------------------------------------------------
st.title("🌧️ Sistema de Alerta Temprana de Inundaciones — Colombia")
st.caption(
    "⚠️ **Proyecto académico** (tesis de Especialización en IA, UNIMINUTO). "
    "No sustituye los avisos oficiales del IDEAM y la UNGRD. "
    "Consulte los **Términos de Uso** al final de la página.  ·  **v1.0 — 2026-06-16**"
)

# --- Selector de FUENTE (pestañas) y MODO ---
col_fuente, col_modo = st.columns([1, 1])
with col_fuente:
    fuente = st.radio("Fuente de datos satelital",
                      ["🛰️ IMERG", "🌧️ CHIRPS"], horizontal=True, index=0,
                      help="IMERG: tiempo real (latencia horas). "
                           "CHIRPS: referencia (latencia ~3 semanas).")
    fuente = "IMERG" if "IMERG" in fuente else "CHIRPS"
with col_modo:
    modo = st.radio("Modo", ["Demo histórica", "Tiempo real"], horizontal=True,
                    index=1,
                    help="Demo: período histórico con inundaciones reales. "
                         "Tiempo real: predicción con datos satelitales recientes.")

# Notas contextuales según selección
if fuente == "CHIRPS" and modo == "Tiempo real":
    st.warning("⚠️ CHIRPS tiene latencia de ~3 semanas. Esta vista es de "
               "**referencia/comparación**, no de alerta temprana operativa. "
               "Para alertas en tiempo real use la fuente IMERG.")
elif fuente == "IMERG" and modo == "Tiempo real":
    st.success("⚡ IMERG en tiempo real (latencia de horas). "
               "IDEAM no disponible en vivo; el modelo usa solo satélite + geografía.")
else:
    st.caption(f"Modelo XGBoost con {fuente} + IDEAM + geografía. "
               "Datos UNGRD 2011–2025.")

# Cargar predicciones de la vista seleccionada
ruta_pred = RUTAS_PRED[(fuente, modo)]
if not ruta_pred.exists():
    st.error(f"No se encontró el archivo de datos para {fuente} / {modo} "
             f"({ruta_pred.name}). Genera las predicciones correspondientes "
             "o elige otra vista.")
    st.stop()

# Carga de predicciones según el modo:
#  - Tiempo real: lee directamente desde GitHub (siempre la última versión que
#    publicó el bot), con respaldo al archivo local si falla la red.
#  - Demo histórica: lee el archivo local (no cambia).
if modo == "Tiempo real" and fuente in URLS_PRED_VIVO:
    _ventana = int(_time.time() // 600)  # cambia cada 10 minutos
    try:
        df = cargar_predicciones_remoto(URLS_PRED_VIVO[fuente], _ventana)
    except Exception:
        # Respaldo: archivo local del contenedor
        _firma = _firma_archivo(ruta_pred)
        df = cargar_predicciones(str(ruta_pred), _firma)
else:
    _firma = _firma_archivo(ruta_pred)
    df = cargar_predicciones(str(ruta_pred), _firma)

geo = cargar_geojson()
rec = cargar_recurrencia()
mens = cargar_mensual()

# Detectar si el modo vivo tiene una sola fecha (oculta el slider)
es_tiempo_real = (modo == "Tiempo real")

# ---------------------------------------------------------------------------
# CONTROLES
# ---------------------------------------------------------------------------
st.sidebar.header("Controles")

# Botón para forzar recarga de datos (limpia el caché sin reiniciar la app)
_ultima_fecha = max(df["fecha"].dt.date.unique())
col_b1, col_b2 = st.sidebar.columns([3, 2])
with col_b1:
    st.caption(f"🛰️ Datos hasta: **{_ultima_fecha.strftime('%d/%m/%Y')}**")
with col_b2:
    if st.button("🔄 Actualizar", help="Recarga los datos más recientes "
                 "publicados por el sistema automático.", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

fechas = sorted(df["fecha"].dt.date.unique())
if len(fechas) == 1:
    # Una sola fecha: mostrarla como info fija
    fecha_sel = fechas[0]
    st.sidebar.info(f"📅 Fecha de pronóstico:\n\n**{fecha_sel.strftime('%d/%m/%Y')}**")
else:
    # Varias fechas (demo o tiempo real con rango): slider
    etiqueta = "Fecha (tiempo real)" if es_tiempo_real else "Fecha"
    fecha_sel = st.sidebar.select_slider(etiqueta, options=fechas,
                                         value=fechas[-1] if es_tiempo_real
                                         else fechas[len(fechas)//2])
horizonte = st.sidebar.radio("Horizonte de pronóstico",
                             ["24h", "48h", "72h"], horizontal=True, index=1)

# Capa del mapa: alertas o precipitación
capa_mapa = st.sidebar.radio("Capa del mapa", ["Alertas", "Precipitación"],
                             horizontal=True,
                             help="Alertas: nivel de riesgo por municipio. "
                                  "Precipitación: lluvia (mm) del día consultado.")
st.sidebar.markdown("---")
st.sidebar.subheader("Vista")
ambito = st.sidebar.radio("Ámbito geográfico", ["🌎 Nacional", "📍 Departamental"])
es_nacional = ambito.startswith("🌎")
depto_sel = None
if not es_nacional:
    depto_sel = st.sidebar.selectbox("Selecciona departamento",
                                     sorted(geo["Depto"].unique()))

# --- Control de Reporte PDF ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Reporte PDF")
alcance_pdf = st.sidebar.radio(
    "Alcance del reporte",
    ["Vista actual", "Nacional completo"],
    help="'Vista actual' exporta lo que estás viendo (nacional o el "
         "departamento seleccionado). 'Nacional completo' exporta todo el país.")
generar_pdf = st.sidebar.button("Generar reporte PDF", use_container_width=True)
st.sidebar.caption("Tras generarlo, el botón de descarga aparece al final "
                   "del dashboard.")

# ---------------------------------------------------------------------------
# FILTRAR DATOS
# ---------------------------------------------------------------------------
col_nivel = f"nivel_{horizonte}"
col_prob  = f"prob_{horizonte}"

dia = df[df["fecha"].dt.date == fecha_sel].copy()
cols_merge = ["cod_dane", col_nivel, col_prob, "hubo_inundacion"]
if "precip_chirps" in dia.columns:
    cols_merge.append("precip_chirps")
dia_geo = geo.merge(dia[cols_merge], on="cod_dane", how="left")
dia_geo[col_nivel] = dia_geo[col_nivel].fillna("baja")

if es_nacional:
    vista_geo = dia_geo
    centro, zoom = [4.6, -74.1], 5
    ambito_txt = "Nacional"
else:
    vista_geo = dia_geo[dia_geo["Depto"] == depto_sel].copy()
    b = vista_geo.total_bounds
    centro = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
    zoom = 7
    ambito_txt = depto_sel

# ---------------------------------------------------------------------------
# MÉTRICAS (distintas según vista)
# ---------------------------------------------------------------------------
n_alta = int((vista_geo[col_nivel] == "alta").sum())
n_media = int((vista_geo[col_nivel] == "media").sum())
n_real = int(vista_geo["hubo_inundacion"].fillna(0).sum())

if es_nacional:
    # Solo fecha + inundaciones reales (alta/media casi no cambian a nivel país)
    c1, c2 = st.columns(2)
    c1.metric("📅 Fecha", fecha_sel.strftime("%d/%m/%Y"))
    c2.metric("💧 Inundaciones reales ese día", n_real)
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Fecha", fecha_sel.strftime("%d/%m/%Y"))
    c2.metric(f"🔴 Alerta ALTA", n_alta)
    c3.metric(f"🟡 Alerta MEDIA", n_media)
    c4.metric("💧 Inundaciones reales", n_real)

# ---------------------------------------------------------------------------
# MAPA + PANEL
# ---------------------------------------------------------------------------
col_mapa, col_panel = st.columns([2, 1])

with col_mapa:
    titulo_capa = "alertas" if capa_mapa == "Alertas" else "precipitación"
    st.subheader(f"Mapa de {titulo_capa} — {horizonte} · {ambito_txt}")
    m = folium.Map(location=centro, zoom_start=zoom, tiles="CartoDB positron")

    tiene_precip = "precip_chirps" in vista_geo.columns

    # Escala de color para precipitación (mm/día)
    def color_precip(mm):
        if mm is None or (isinstance(mm, float) and pd.isna(mm)):
            return "#f0f0f0"
        if mm < 1:    return "#f7fbff"
        if mm < 5:    return "#c6dbef"
        if mm < 10:   return "#6baed6"
        if mm < 20:   return "#3182bd"
        if mm < 40:   return "#08519c"
        return "#08306b"

    if capa_mapa == "Alertas":
        def estilo(feat):
            nivel = feat["properties"].get(col_nivel, "baja")
            return {"fillColor": COLORES.get(nivel, "#cccccc"),
                    "color": "#999999", "weight": 0.15, "fillOpacity": 0.7,
                    "opacity": 0.4}
        campos = ["MpNombre", "Depto", col_nivel, col_prob]
        alias = ["Municipio:", "Depto:", "Alerta:", "Prob:"]
        if tiene_precip:
            campos.append("precip_chirps"); alias.append("Lluvia (mm):")
        folium.GeoJson(vista_geo.to_json(), style_function=estilo,
            tooltip=folium.GeoJsonTooltip(fields=campos, aliases=alias,
                                          localize=True)).add_to(m)
        m.get_root().html.add_child(Element(LEYENDA_HTML))

    else:  # Capa de Precipitación: relleno por lluvia, municipios con borde
        def estilo(feat):
            mm = feat["properties"].get("precip_chirps")
            return {"fillColor": color_precip(mm),
                    "color": "#999999", "weight": 0.15, "fillOpacity": 0.78,
                    "opacity": 0.4}
        campos = ["MpNombre", "Depto"]
        alias = ["Municipio:", "Depto:"]
        if tiene_precip:
            campos.append("precip_chirps"); alias.append("Lluvia (mm):")
        folium.GeoJson(vista_geo.to_json(), style_function=estilo,
            tooltip=folium.GeoJsonTooltip(fields=campos, aliases=alias,
                                          localize=True)).add_to(m)
        m.get_root().html.add_child(Element(LEYENDA_PRECIP))

    # Marcadores de inundaciones reales (en ambas capas)
    reales = vista_geo[vista_geo["hubo_inundacion"] == 1]
    for _, r in reales.iterrows():
        c = r["geometry"].centroid
        folium.CircleMarker([c.y, c.x], radius=4, color="#ffffff",
                            weight=1.2, fill=True, fill_color="#ff1493",
                            fill_opacity=1.0,
                            tooltip=f"💧 Inundación real: {r['MpNombre']}").add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])
    if capa_mapa == "Precipitación" and not tiene_precip:
        st.info("Este conjunto de datos no incluye la columna de precipitación. "
                "Regenera las predicciones con la versión actualizada de los scripts.")
    if es_tiempo_real:
        st.caption("En modo tiempo real no se muestran inundaciones reales "
                   "(el evento aún no ha ocurrido / no hay reporte UNGRD todavía).")
    elif len(reales) > 0:
        st.caption("🔴 Los puntos rojos marcan inundaciones reportadas ese día (UNGRD).")

with col_panel:
    if es_nacional:
        st.subheader("Confiabilidad por región")
        st.caption("Probabilidad media de alerta por región natural "
                   "(el modelo es más confiable a escala regional).")
        resumen_reg = (dia_geo.groupby("region")[col_prob]
                       .mean().sort_values(ascending=False).reset_index())
        resumen_reg.columns = ["Región", "Prob. media"]
        resumen_reg = resumen_reg[resumen_reg["Región"] != "Sin región"]
        st.dataframe(resumen_reg, hide_index=True, use_container_width=True,
                     column_config={"Prob. media": st.column_config.ProgressColumn(
                         "Prob. media de alerta", min_value=0,
                         max_value=float(resumen_reg["Prob. media"].max() or 1),
                         format="%.3f")})
    else:
        st.subheader(f"Resumen — {depto_sel}")
        st.metric("Probabilidad media de alerta", f"{vista_geo[col_prob].mean():.3f}")

    st.subheader("🔴 Municipios en alerta ALTA")
    altas = (vista_geo[vista_geo[col_nivel] == "alta"]
             .sort_values(col_prob, ascending=False)
             [["MpNombre", "Depto", col_prob]])
    altas.columns = ["Municipio", "Departamento", "Probabilidad"]
    if len(altas):
        st.dataframe(altas, hide_index=True, use_container_width=True, height=240)
    else:
        st.info("Sin municipios en alerta alta en este ámbito/fecha.")

# ===========================================================================
# SECCIÓN INFERIOR
# ===========================================================================
st.markdown("---")

if es_nacional:
    # ---- Distribución de alertas (solo alta/media, top 10 deptos) ----
    st.subheader("Distribución de alertas (alta y media)")
    st.caption("Municipios en alerta alta/media — varía con la fecha y el horizonte.")

    def grafica_alertas(data, campo, titulo, top=None):
        d = data[data[col_nivel].isin(["alta", "media"])]
        if len(d) == 0:
            return None
        conteo = d.groupby([campo, col_nivel]).size().reset_index(name="municipios")
        if top:
            tot = conteo.groupby(campo)["municipios"].sum().nlargest(top).index
            conteo = conteo[conteo[campo].isin(tot)]
        # Altura basada en el nº REAL de categorías mostradas (más espacio por barra)
        n_cat = conteo[campo].nunique()
        altura = max(220, 40 * n_cat)
        return (alt.Chart(conteo).mark_bar().encode(
            x=alt.X("municipios:Q", title="Nº municipios en alerta"),
            y=alt.Y(f"{campo}:N", title=None, sort="-x",
                    axis=alt.Axis(labelLimit=200, labelFontSize=12)),
            color=alt.Color(f"{col_nivel}:N",
                            scale=alt.Scale(domain=["alta", "media"],
                                            range=[COLORES["alta"], COLORES["media"]]),
                            legend=alt.Legend(title="Nivel")),
            order=alt.Order(f"{col_nivel}:N"),
            tooltip=[campo, col_nivel, "municipios"]
        ).properties(height=altura, title=titulo))

    g1, g2 = st.columns(2)
    with g1:
        ch = grafica_alertas(dia_geo, "region", "Por región natural")
        if ch: st.altair_chart(ch, use_container_width=True)
        else: st.info("Sin alertas alta/media en esta fecha.")
    with g2:
        ch = grafica_alertas(dia_geo, "Depto", "Por departamento (top 10)", top=10)
        if ch: st.altair_chart(ch, use_container_width=True)
        else: st.info("Sin alertas alta/media en esta fecha.")

    # ---- Recurrencia nacional top 10 + estacionalidad mensual ----
    st.markdown("---")
    st.subheader("Municipios más recurrentes (histórico real, UNGRD 2011-2025)")

    if rec is not None:
        rec_n = rec.merge(geo[["cod_dane", "MpNombre", "Depto"]], on="cod_dane", how="left")
        top10 = rec_n.sort_values("dias_inundacion_real", ascending=False).head(10)

        ch = (alt.Chart(top10).mark_bar(color="#0033cc").encode(
            x=alt.X("dias_inundacion_real:Q", title="Días con inundación (2011-2025)"),
            y=alt.Y("MpNombre:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200, labelFontSize=12)),
            tooltip=["MpNombre", "Depto", "dias_inundacion_real"]
        ).properties(height=320, title="Top 10 municipios con más inundaciones (Colombia)"))
        st.altair_chart(ch, use_container_width=True)

        # Estacionalidad mensual de esos top 10
        if mens is not None:
            st.subheader("Estacionalidad: ¿en qué meses se inundan más?")
            st.caption("Distribución mensual de inundaciones de los 10 municipios más recurrentes.")
            mens_top = mens[mens["cod_dane"].isin(top10["cod_dane"])].merge(
                geo[["cod_dane", "MpNombre"]], on="cod_dane", how="left")
            mens_top["mes_nombre"] = mens_top["mes"].map(MESES)
            heatmap = (alt.Chart(mens_top).mark_rect().encode(
                x=alt.X("mes_nombre:N", title="Mes",
                        sort=list(MESES.values())),
                y=alt.Y("MpNombre:N", title=None,
                        axis=alt.Axis(labelLimit=200, labelFontSize=12)),
                color=alt.Color("eventos:Q", scale=alt.Scale(scheme="reds"),
                                legend=alt.Legend(title="Eventos")),
                tooltip=["MpNombre", "mes_nombre", "eventos"]
            ).properties(height=max(360, 36 * mens_top["MpNombre"].nunique()),
                         title="Mapa de calor: inundaciones por mes"))
            st.altair_chart(heatmap, use_container_width=True)

    # ---- Otras gráficas estadísticas de interés ----
    st.markdown("---")
    st.subheader("Estadísticas generales del histórico")
    ge1, ge2 = st.columns(2)
    with ge1:
        # Inundaciones por mes (todo el país, todos los años)
        if mens is not None:
            por_mes = mens.groupby("mes")["eventos"].sum().reset_index()
            por_mes["mes_nombre"] = por_mes["mes"].map(MESES)
            ch = (alt.Chart(por_mes).mark_bar(color="#4575b4").encode(
                x=alt.X("mes_nombre:N", title="Mes", sort=list(MESES.values())),
                y=alt.Y("eventos:Q", title="Total inundaciones"),
                tooltip=["mes_nombre", "eventos"]
            ).properties(height=300, title="Inundaciones por mes (nacional, 2011-2025)"))
            st.altair_chart(ch, use_container_width=True)
    with ge2:
        # Inundaciones por región
        if rec is not None:
            rec_reg = rec.merge(geo[["cod_dane", "region"]], on="cod_dane", how="left")
            por_reg = (rec_reg.groupby("region")["dias_inundacion_real"].sum()
                       .reset_index().sort_values("dias_inundacion_real", ascending=False))
            por_reg = por_reg[por_reg["region"] != "Sin región"]
            ch = (alt.Chart(por_reg).mark_bar(color="#91bfdb").encode(
                x=alt.X("dias_inundacion_real:Q", title="Total inundaciones"),
                y=alt.Y("region:N", title=None, sort="-x"),
                tooltip=["region", "dias_inundacion_real"]
            ).properties(height=300, title="Inundaciones por región natural (2011-2025)"))
            st.altair_chart(ch, use_container_width=True)

else:
    # ---- Vista departamental: recurrencia predicho vs real (top 10) ----
    st.subheader(f"Recurrencia de alertas — {depto_sel}")
    st.caption("Municipios más recurrentes en alertas del modelo (demo) vs su "
               "historial real (UNGRD 2011-2025).")

    if rec is not None:
        cods_depto = vista_geo["cod_dane"].unique()
        rec_d = rec[rec["cod_dane"].isin(cods_depto)].merge(
            geo[["cod_dane", "MpNombre"]], on="cod_dane", how="left")

        gr1, gr2 = st.columns(2)
        with gr1:
            top_pred = rec_d.sort_values("dias_alerta_pred", ascending=False).head(10)
            ch = (alt.Chart(top_pred).mark_bar(color=COLORES["alta"]).encode(
                x=alt.X("dias_alerta_pred:Q", title="Días en alerta (modelo, demo)"),
                y=alt.Y("MpNombre:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200, labelFontSize=12)),
                tooltip=["MpNombre", "dias_alerta_pred"]
            ).properties(height=350, title="Predicho (modelo) — top 10"))
            st.altair_chart(ch, use_container_width=True)
        with gr2:
            top_real = rec_d.sort_values("dias_inundacion_real", ascending=False).head(10)
            ch = (alt.Chart(top_real).mark_bar(color="#0033cc").encode(
                x=alt.X("dias_inundacion_real:Q", title="Días con inundación (histórico)"),
                y=alt.Y("MpNombre:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200, labelFontSize=12)),
                tooltip=["MpNombre", "dias_inundacion_real"]
            ).properties(height=350, title="Real (UNGRD 2011-2025) — top 10"))
            st.altair_chart(ch, use_container_width=True)

        # Estacionalidad del departamento
        if mens is not None:
            st.subheader(f"Estacionalidad de inundaciones — {depto_sel}")
            mens_d = mens[mens["cod_dane"].isin(cods_depto)]
            por_mes_d = mens_d.groupby("mes")["eventos"].sum().reset_index()
            por_mes_d["mes_nombre"] = por_mes_d["mes"].map(MESES)
            ch = (alt.Chart(por_mes_d).mark_bar(color="#4575b4").encode(
                x=alt.X("mes_nombre:N", title="Mes", sort=list(MESES.values())),
                y=alt.Y("eventos:Q", title="Inundaciones"),
                tooltip=["mes_nombre", "eventos"]
            ).properties(height=280, title=f"Inundaciones por mes en {depto_sel} (2011-2025)"))
            st.altair_chart(ch, use_container_width=True)

# ---------------------------------------------------------------------------
# REPORTE PDF (generación bajo demanda)
# ---------------------------------------------------------------------------
if generar_pdf:
    with st.spinner("Generando reporte PDF..."):
        if alcance_pdf == "Nacional completo":
            geo_reporte = dia_geo
            ambito_reporte = "Nacional"
        else:
            geo_reporte = vista_geo
            ambito_reporte = ambito_txt
        # En reporte nacional, es_nacional=True para las gráficas
        es_nac_reporte = (alcance_pdf == "Nacional completo") or es_nacional
        try:
            pdf_bytes = generar_reporte(
                geo_reporte, dia_geo, col_nivel, col_prob,
                fecha_sel, horizonte, ambito_reporte, es_nac_reporte,
                fuente=fuente, modo=modo, rec=rec, mens=mens, geo=geo)
            st.markdown("---")
            st.success("✅ Reporte generado. Descárgalo aquí:")
            nombre = (f"reporte_SAT_{ambito_reporte.replace(' ', '_')}_"
                      f"{fecha_sel.strftime('%Y%m%d')}_{horizonte}.pdf")
            st.download_button("⬇️ Descargar reporte PDF", data=pdf_bytes,
                               file_name=nombre, mime="application/pdf",
                               use_container_width=False)
        except Exception as e:
            st.error(f"No se pudo generar el PDF: {e}")

# ---------------------------------------------------------------------------
# PIE — TÉRMINOS DE USO
# ---------------------------------------------------------------------------
st.markdown("---")

with st.expander("📋 Términos de Uso  ·  v1.0 — 2026-06-16", expanded=False):
    st.markdown(
        """
**Sistema de Alerta Temprana de Inundaciones para Colombia basado en Inteligencia Artificial**
**Versión 1.0 — 16 de junio de 2026**

---

#### 1. Naturaleza del proyecto
Este sistema es un **proyecto académico de tesis**, desarrollado como trabajo de grado
para optar al título de Especialista en Inteligencia Artificial de la **Corporación
Universitaria Minuto de Dios (UNIMINUTO)**. Tiene fines investigativos, educativos y
demostrativos. **No constituye un servicio oficial de alerta ni un producto operativo
de ninguna entidad pública.**

#### 2. Alcance
El sistema estima, mediante un modelo de inteligencia artificial entrenado con datos
abiertos (precipitación satelital CHIRPS e IMERG, estaciones del IDEAM y registros
históricos de la UNGRD), la probabilidad de ocurrencia de inundaciones a escala
municipal para horizontes de 24, 48 y 72 horas. Está diseñado para:

- Servir como herramienta de **apoyo** a la consulta y priorización del riesgo de inundación.
- Ofrecer una capa de información anticipada de **acceso libre**, especialmente útil
  para municipios que carecen de sistemas de alerta propios.
- Funcionar a escala **regional y semanal**, que es la escala en la que el modelo
  alcanza su mayor confiabilidad.

#### 3. Limitaciones
El usuario reconoce y acepta que:

- Las alertas son **estimaciones probabilísticas, no certezas**. Una alerta no garantiza
  que ocurra una inundación, ni su ausencia garantiza que no ocurra.
- La confiabilidad del sistema es de carácter **regional y semanal**, y no debe
  interpretarse como una predicción exacta de municipio y día.
- El sistema **no pronostica** la magnitud, extensión, profundidad ni duración de una
  inundación, ni emite alertas de crecientes súbitas con anticipación de minutos.
- El modelo puede **subestimar eventos convectivos muy localizados** y hereda los sesgos
  de sus fuentes de datos (subreporte histórico, cobertura parcial de estaciones).
- En modo de operación en tiempo real, el sistema utiliza únicamente datos satelitales y
  geomorfológicos; las estaciones del IDEAM no se incorporan en vivo en esta versión.
- El funcionamiento depende de la disponibilidad de servicios de terceros (Google Earth
  Engine, Streamlit Cloud, GitHub) y de la conexión a internet del usuario.

#### 4. Responsabilidad
- **No sustituye los avisos oficiales del IDEAM ni de la UNGRD.** Ante cualquier situación
  de riesgo, las fuentes oficiales y las autoridades competentes de gestión del riesgo
  prevalecen siempre.
- El sistema es una herramienta de apoyo a la decisión y **no reemplaza el juicio
  profesional, el monitoreo local ni la autoridad** de los consejos municipales y
  departamentales de gestión del riesgo de desastres.
- El autor **no se hace responsable** por decisiones, acciones u omisiones tomadas con
  base en la información del sistema, ni por daños directos o indirectos derivados de su
  uso o de su indisponibilidad. La información se ofrece "tal cual", sin garantías de
  ningún tipo.
- El usuario emplea el sistema bajo su propia responsabilidad.

#### 5. Uso de datos y privacidad
El sistema se construyó exclusivamente con **datos públicos y abiertos**, sin información
personal ni datos sensibles de individuos. La variable de inundación opera a escala
municipal, no individual. El código fuente es abierto y reproducible, lo que permite
auditar el funcionamiento del sistema.

#### 6. Citación
Este es un trabajo académico. Si utiliza, referencia o se apoya en este sistema o sus
resultados, debe citarlo de la siguiente manera (APA 7):

> Alpala Aguilar, J. A. (2026). *Sistema de alerta temprana de inundaciones para Colombia
> basado en inteligencia artificial* [Monografía de especialización, Corporación
> Universitaria Minuto de Dios]. UNIMINUTO.

#### 7. Recursos del proyecto
- **Repositorio de código (GitHub):** https://github.com/jorgealpala/sat-inundaciones-colombia.git
- **Conjunto de datos (Zenodo):** https://zenodo.org/records/20713063

#### 8. Versión
**Versión 1.0** — 16 de junio de 2026. Este es un sistema en evolución; las metodologías,
datos y resultados pueden actualizarse en versiones posteriores.

---

**Autor:** Jorge Armando Alpala Aguilar — Ingeniero Civil, Especialista en Sistemas de
Información Geográfica, Magíster en Geomática.
**Contacto:** jorge.alpala.1987@gmail.com
        """
    )

st.caption(
    "Sistema de Alerta Temprana de Inundaciones — Colombia  ·  v1.0 (2026-06-16)  ·  "
    "Proyecto académico — UNIMINUTO  ·  No sustituye los avisos oficiales del IDEAM y la UNGRD.  ·  "
    "© 2026 Jorge A. Alpala Aguilar"
)
