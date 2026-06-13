"""
============================================================================
 reporte_pdf.py  —  Generación de reporte PDF para el dashboard SAT  (v2)
 ----------------------------------------------------------------------------
 Reporte PDF ampliado con:
   - Encabezado (fuente, modo, fecha, horizonte, ámbito)
   - Tabla resumen del día
   - Mapa de alertas (estático) + mapa de precipitación (estático)
   - Tabla de municipios en alerta alta
   - Gráfica de distribución de alertas por departamento (alta/media)
   - Gráfica de probabilidad media por región
   - Recurrencia histórica (top 10) y estacionalidad mensual (si hay datos)

 Uso desde app.py:
   from reporte_pdf import generar_reporte
   pdf = generar_reporte(vista_geo, dia_geo, col_nivel, col_prob, fecha,
                         horizonte, ambito_txt, es_nacional,
                         fuente=..., modo=..., rec=..., mens=..., geo=...)
 Devuelve bytes para st.download_button.
============================================================================
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

COLORES = {"alta": "#d73027", "media": "#fee08b", "baja": "#1a9850"}
MESES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
         7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}


def _fig_to_image(fig, ancho_cm=16):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = ancho_cm * cm
    img.drawHeight = ancho_cm * cm * ratio
    return img


def _mapa_estatico(vista_geo, col_nivel, titulo):
    fig, ax = plt.subplots(figsize=(7, 8))
    for nivel, color in COLORES.items():
        sub = vista_geo[vista_geo[col_nivel] == nivel]
        if len(sub):
            sub.plot(ax=ax, color=color, edgecolor="#999999", linewidth=0.15)
    reales = vista_geo[vista_geo["hubo_inundacion"] == 1]
    if len(reales):
        reales.geometry.centroid.plot(ax=ax, color="#ff1493", markersize=18,
                                      marker="o", zorder=5)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.axis("off")
    leyenda = [Patch(facecolor=COLORES["alta"], label="Alta"),
               Patch(facecolor=COLORES["media"], label="Media"),
               Patch(facecolor=COLORES["baja"], label="Baja"),
               plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#ff1493", markersize=8,
                          label="Inundación real")]
    ax.legend(handles=leyenda, loc="lower left", fontsize=9, framealpha=0.9)
    return fig


def _mapa_precip(vista_geo, titulo):
    """Mapa de precipitación (mm/día) por municipio."""
    if "precip_chirps" not in vista_geo.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 8))
    # escala discreta de azules
    bins = [0, 1, 5, 10, 20, 40, 1e9]
    cols = ["#f7fbff", "#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b"]
    for i in range(len(bins) - 1):
        sub = vista_geo[(vista_geo["precip_chirps"] >= bins[i]) &
                        (vista_geo["precip_chirps"] < bins[i + 1])]
        if len(sub):
            sub.plot(ax=ax, color=cols[i], edgecolor="#bbbbbb", linewidth=0.1)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.axis("off")
    etiquetas = ["<1", "1–5", "5–10", "10–20", "20–40", ">40"]
    leyenda = [Patch(facecolor=cols[i], label=etiquetas[i]) for i in range(6)]
    ax.legend(handles=leyenda, loc="lower left", fontsize=8,
              title="mm/día", framealpha=0.9)
    return fig


def _grafica_region(dia_geo, col_prob):
    resumen = (dia_geo.groupby("region")[col_prob].mean()
               .sort_values(ascending=False))
    resumen = resumen[resumen.index != "Sin región"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(resumen.index, resumen.values, color="#4575b4")
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad media de alerta")
    ax.set_title("Probabilidad media por región natural", fontsize=12,
                 fontweight="bold")
    for i, v in enumerate(resumen.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    return fig


def _grafica_depto(dia_geo, col_nivel):
    """Top 10 departamentos por nº de municipios en alerta alta/media."""
    d = dia_geo[dia_geo[col_nivel].isin(["alta", "media"])]
    if len(d) == 0:
        return None
    conteo = d.groupby(["Depto", col_nivel]).size().unstack(fill_value=0)
    conteo["tot"] = conteo.sum(axis=1)
    conteo = conteo.sort_values("tot", ascending=False).head(10).drop(columns="tot")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    izq = np.zeros(len(conteo))
    for nivel in ["alta", "media"]:
        if nivel in conteo.columns:
            ax.barh(conteo.index, conteo[nivel], left=izq,
                    color=COLORES[nivel], label=nivel.capitalize())
            izq += conteo[nivel].values
    ax.invert_yaxis()
    ax.set_xlabel("Nº municipios en alerta")
    ax.set_title("Top 10 departamentos en alerta (alta/media)", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    return fig


def _grafica_recurrencia(rec, geo, depto=None):
    """Top 10 municipios por recurrencia real histórica."""
    if rec is None:
        return None
    r = rec.merge(geo[["cod_dane", "MpNombre", "Depto"]], on="cod_dane", how="left")
    if depto:
        r = r[r["Depto"] == depto]
    r = r.sort_values("dias_inundacion_real", ascending=False).head(10)
    if len(r) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(r["MpNombre"], r["dias_inundacion_real"], color="#0033cc")
    ax.invert_yaxis()
    ax.set_xlabel("Días con inundación (2011–2025)")
    titulo = f"Top 10 municipios más recurrentes" + (f" — {depto}" if depto else "")
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    return fig


def _grafica_estacionalidad(mens, geo, depto=None):
    """Inundaciones por mes (nacional o de un departamento)."""
    if mens is None:
        return None
    m = mens.copy()
    if depto:
        cods = geo[geo["Depto"] == depto]["cod_dane"].unique()
        m = m[m["cod_dane"].isin(cods)]
    por_mes = m.groupby("mes")["eventos"].sum().reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar([MESES[i] for i in range(1, 13)], por_mes.values, color="#4575b4")
    ax.set_ylabel("Inundaciones")
    titulo = "Estacionalidad de inundaciones (2011–2025)" + (f" — {depto}" if depto else "")
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    return fig


def generar_reporte(vista_geo, dia_geo, col_nivel, col_prob,
                    fecha, horizonte, ambito_txt, es_nacional,
                    fuente="IMERG", modo="Tiempo real",
                    rec=None, mens=None, geo=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle("titulo", parent=styles["Title"],
                                  fontSize=18, textColor=colors.HexColor("#1a3a5c"))
    sub_style = ParagraphStyle("sub", parent=styles["Heading2"],
                               fontSize=13, textColor=colors.HexColor("#2c5f8a"))
    story = []

    # --- Encabezado ---
    story.append(Paragraph("Sistema de Alerta Temprana de Inundaciones — Colombia",
                           titulo_style))
    story.append(Paragraph(
        f"Reporte de alertas · {ambito_txt} · {fuente} · {modo} · Horizonte {horizonte}",
        sub_style))
    story.append(Paragraph(
        f"Fecha de pronóstico: <b>{fecha.strftime('%d/%m/%Y')}</b> &nbsp;|&nbsp; "
        f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- Mapa de alertas ---
    story.append(Paragraph("Mapa de alertas", sub_style))
    fig_mapa = _mapa_estatico(vista_geo, col_nivel,
                              f"Alertas {horizonte} — {ambito_txt}")
    story.append(_fig_to_image(fig_mapa, ancho_cm=12))

    # --- Mapa de precipitación ---
    fig_precip = _mapa_precip(vista_geo, f"Precipitación (mm/día) — {ambito_txt}")
    if fig_precip is not None:
        story.append(PageBreak())
        story.append(Paragraph("Mapa de precipitación", sub_style))
        story.append(_fig_to_image(fig_precip, ancho_cm=12))

    # --- Tabla de alertas altas ---
    story.append(PageBreak())
    story.append(Paragraph("Municipios en alerta ALTA", sub_style))
    altas = (vista_geo[vista_geo[col_nivel] == "alta"]
             .sort_values(col_prob, ascending=False)
             [["MpNombre", "Depto", col_prob]].head(25))
    if len(altas):
        data = [["Municipio", "Departamento", "Probabilidad"]]
        for _, r in altas.iterrows():
            data.append([str(r["MpNombre"]), str(r["Depto"]), f"{r[col_prob]:.3f}"])
        t2 = Table(data, colWidths=[6 * cm, 5 * cm, 3 * cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d73027")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#fbeae8")]),
        ]))
        story.append(t2)
    else:
        story.append(Paragraph("Sin municipios en alerta alta en este ámbito/fecha.",
                               styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    # --- Distribución por departamento ---
    fig_depto = _grafica_depto(dia_geo if es_nacional else vista_geo, col_nivel)
    if fig_depto is not None:
        story.append(Paragraph("Distribución de alertas por departamento", sub_style))
        story.append(_fig_to_image(fig_depto, ancho_cm=15))
        story.append(Spacer(1, 0.4 * cm))

    # --- Probabilidad por región ---
    story.append(Paragraph("Probabilidad media por región", sub_style))
    fig_reg = _grafica_region(dia_geo, col_prob)
    story.append(_fig_to_image(fig_reg, ancho_cm=15))

    # --- Recurrencia + estacionalidad (si hay datos) ---
    if rec is not None and geo is not None:
        story.append(PageBreak())
        story.append(Paragraph("Recurrencia histórica de inundaciones", sub_style))
        depto = None if es_nacional else ambito_txt
        fig_rec = _grafica_recurrencia(rec, geo, depto)
        if fig_rec is not None:
            story.append(_fig_to_image(fig_rec, ancho_cm=15))
            story.append(Spacer(1, 0.4 * cm))
        fig_est = _grafica_estacionalidad(mens, geo, depto)
        if fig_est is not None:
            story.append(_fig_to_image(fig_est, ancho_cm=15))

    # --- Pie ---
    story.append(Spacer(1, 0.6 * cm))
    pie_style = ParagraphStyle("pie", parent=styles["Normal"], fontSize=8,
                               textColor=colors.grey)
    story.append(Paragraph(
        f"Sistema demostrativo académico (tesis de especialización en IA). "
        f"Modelo XGBoost basado en {fuente} + IDEAM + geografía, datos UNGRD "
        "2011–2025. Las alertas son estimaciones probabilísticas y no "
        "sustituyen los avisos oficiales del IDEAM y la UNGRD.", pie_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
