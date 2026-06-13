# Guía de publicación — SAT de Inundaciones Colombia

Esta guía te lleva paso a paso para publicar el dashboard en internet con
actualización automática 2 veces al día.

---

## ESTRUCTURA DEL REPOSITORIO

Organiza tu repositorio de GitHub así:

```
sat-inundaciones-colombia/
├── dashboard_app/
│   ├── app.py
│   ├── reporte_pdf.py
│   └── data/
│       ├── predicciones_demo.parquet           (CHIRPS demo)
│       ├── predicciones_demo_imerg.parquet      (IMERG demo)
│       ├── predicciones_vivo.parquet            (CHIRPS vivo, opcional)
│       ├── predicciones_vivo_imerg.parquet      (IMERG vivo — se actualiza solo)
│       ├── municipios_dashboard.geojson
│       ├── recurrencia_municipios.parquet
│       └── recurrencia_mensual.parquet
├── scripts/
│   ├── 23_prediccion_vivo_imerg.py              (lo usa GitHub Actions)
│   ├── geografia_municipios.csv
│   └── modelos_imerg/
│       ├── modelo_imerg_alerta_24h.pkl
│       ├── modelo_imerg_alerta_48h.pkl
│       └── modelo_imerg_alerta_72h.pkl
├── .github/
│   └── workflows/
│       └── actualizar.yml
├── requirements.txt
├── .gitignore
└── README.md

IMPORTANTE: gee-key.json NO se sube (está en .gitignore). Va como secret.
```

---

## PASO 1 — Crear el repositorio en GitHub

1. Crea una cuenta en github.com si no tienes.
2. Crea un repositorio nuevo (ej. `sat-inundaciones-colombia`), público.
3. Sube los archivos con la estructura de arriba. Puedes usar:
   - La interfaz web (arrastrar archivos), o
   - Git desde tu PC:
     ```
     git init
     git add .
     git commit -m "Versión inicial del SAT"
     git branch -M main
     git remote add origin https://github.com/TU_USUARIO/sat-inundaciones-colombia.git
     git push -u origin main
     ```

VERIFICA: que gee-key.json NO aparezca en el repo (debe estar ignorado).

---

## PASO 2 — Configurar los secrets en GitHub

Los secrets guardan la credencial GEE de forma segura (no visible en el código).

1. En tu repo: **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**. Crea dos:

   **Secret 1:**
   - Name: `GEE_KEY_JSON`
   - Value: pega TODO el contenido de tu gee-key.json (el JSON completo)

   **Secret 2:**
   - Name: `GEE_PROJECT`
   - Value: `geovolcanes-scr-piloto`

---

## PASO 3 — Activar GitHub Actions

1. El archivo `.github/workflows/actualizar.yml` ya define el proceso.
2. En tu repo: pestaña **Actions**. Si pide habilitar workflows, acepta.
3. Para probarlo YA (sin esperar al horario):
   - Entra a **Actions** → "Actualizar predicciones SAT" → **Run workflow**
4. Verás el progreso. Si todo va bien, al terminar habrá un commit nuevo
   con el parquet actualizado.

Horario configurado: 5:00 AM y 5:00 PM hora Colombia (puedes cambiarlo en el
cron del archivo .yml).

---

## PASO 4 — Desplegar en Streamlit Cloud

1. Ve a https://share.streamlit.io/ e inicia sesión con tu cuenta de GitHub.
2. Click **New app** → **From existing repo**.
3. Configura:
   - Repository: `TU_USUARIO/sat-inundaciones-colombia`
   - Branch: `main`
   - Main file path: `dashboard_app/app.py`
4. Click **Deploy**. Espera unos minutos (instala dependencias).
5. Tendrás una URL pública tipo:
   `https://sat-inundaciones-colombia.streamlit.app`

NOTA: el dashboard publicado solo LEE los parquet, no consulta GEE. Por eso
carga rápido. La actualización la hace GitHub Actions por detrás.

---

## CÓMO FUNCIONA EL CICLO COMPLETO

```
2 veces al día:
  GitHub Actions corre el script 23 → consulta GEE → actualiza el parquet
  → hace commit → Streamlit Cloud detecta el cambio → recarga datos frescos

El usuario:
  Abre la URL → ve las predicciones más recientes al instante
```

Cuando llegue agosto/septiembre, el dashboard mostrará automáticamente esas
fechas, porque GitHub Actions habrá ido actualizando el parquet día a día.

---

## VERIFICACIÓN FINAL

- [ ] El repo NO contiene gee-key.json
- [ ] Los dos secrets están configurados
- [ ] El workflow corre sin error (pestaña Actions)
- [ ] El dashboard abre en su URL pública
- [ ] Al correr el workflow manualmente, el parquet se actualiza

---

## SEGURIDAD — RECORDATORIO

- La credencial gee-key.json va SOLO como secret, nunca en el código.
- Si la credencial se expuso antes, revócala y genera una nueva en Google
  Cloud Console, y actualiza el secret GEE_KEY_JSON.
