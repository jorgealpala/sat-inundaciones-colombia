# 🌧️ Sistema de Alerta Temprana de Inundaciones — Colombia

Sistema de alerta temprana (SAT) de inundaciones a nivel municipal para
Colombia, basado en aprendizaje automático (XGBoost) y precipitación satelital
en tiempo real.

## Descripción

El sistema predice el riesgo de inundación por municipio en horizontes de 24,
48 y 72 horas, combinando:

- **Precipitación satelital**: GPM IMERG (tiempo real, latencia de horas) y
  CHIRPS (referencia histórica).
- **Estaciones terrestres**: IDEAM (datos históricos de precipitación).
- **Variables geográficas**: elevación, pendiente, acumulación de flujo,
  índice topográfico de humedad, ocurrencia de agua.
- **Histórico de eventos**: base de datos de inundaciones de la UNGRD
  (2011–2025).

## Características

- Mapa interactivo de alertas por municipio (niveles: alta, media, baja).
- Navegación nacional y departamental.
- Predicción en tiempo real con actualización automática 2 veces al día.
- Comparación entre fuentes satelitales (IMERG vs CHIRPS).
- Gráficas de distribución regional, recurrencia histórica y estacionalidad.
- Generación de reportes en PDF.

## Hallazgo metodológico central

El modelo alcanza su máxima confiabilidad a escala **región-semana**
(F1 ≈ 0.77), que corresponde a la escala espacio-temporal en que un sistema de
alerta de inundaciones resulta operativamente útil, más que a la predicción
exacta de municipio-día.

## Arquitectura

```
GitHub Actions (2x/día) → consulta GEE → genera predicciones → commit
                                                                   │
                                                                   ▼
Streamlit Cloud ← lee predicciones (rápido) ← repositorio actualizado
```

## Tecnologías

Python · XGBoost · scikit-learn · Google Earth Engine · Streamlit · GeoPandas ·
Folium · GitHub Actions

## Aviso

Sistema desarrollado con fines académicos (tesis de especialización en
Inteligencia Artificial). Las alertas son estimaciones probabilísticas y no
sustituyen los avisos oficiales del IDEAM y la UNGRD.
