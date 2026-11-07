import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import time
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# 1. Configuración de página
st.set_page_config(
    page_title="DSP & Vision Lab | Telemetría Foliar UNAD",
    page_icon="📡",
    layout="wide"
)

# 2. Inyección CSS: Dashboard de Ingeniería de Telecomunicaciones & DSP
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    code, .stMetric, .telemetry-val {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Banner Superior Estilo Rack / Telemetría */
    .telecom-banner {
        background: linear-gradient(135deg, #0a0f1d 0%, #0d1b2a 50%, #1b263b 100%);
        border: 1px solid #1f3a5f;
        border-left: 6px solid #00f5d4;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 245, 212, 0.08);
    }

    .telecom-title {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .telecom-sub {
        color: #00b4d8;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 6px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .telecom-badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }

    .badge-chip {
        background: rgba(0, 180, 216, 0.12);
        color: #90e0ef;
        border: 1px solid rgba(0, 180, 216, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Caja de Diagnóstico Principal - Tipo Instrumento Digital */
    .verdict-box {
        background: #0b1320;
        border: 1px solid #233554;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }

    .verdict-header {
        font-size: 0.75rem;
        color: #64748b;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 700;
    }

    .verdict-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 6px 0;
    }

    /* Canales de Detección */
    .channel-row {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Técnico - Nodo de Control
with st.sidebar:
    st.markdown("### 🛰️ NODO DE CONTROL SIPROINT")
    st.caption("UNAD · Escuela de Ciencias Básicas, Tecnología e Ingeniería (ECBTI)")
    st.markdown("---")
    
    st.markdown("**📡 ENTORNO DE TELECOMUNICACIONES**")
    st.markdown("- **Estación:** Pamplona, N. de S.")
    st.markdown("- **Topología:** Muestreo a Campo Abierto")
    st.markdown("- **Motor DSP:** Segmentación HSV + GLCM")
    st.markdown("- **Clasificador:** KNN (K-Nearest Neighbors)")
    st.markdown("- **Pipeline:** 11 Descriptores Espaciales")
    
    st.markdown("---")
    st.markdown("**📊 ESPECTRO FITOSANITARIO**")
    st.markdown("""
    `CH_01` 🟢 **Sana**  
    `CH_02` 🟣 **Viruela** *(R. tulasnei)*  
    `CH_03` 🟤 **Antracnosis** *(Colletotrichum)*  
    `CH_04` ⚪ **Oídio** *(P. aphanis)*  
    `CH_05` ❓ **No reconocida / Dudosa**
    """)
    st.markdown("---")
    st.caption("SYS_STATUS: ONLINE | INTERFACE: STREAMLIT CORE")

# 4. Banner Superior de Ingeniería
st.markdown("""
    <div class="telecom-banner">
        <div class="telecom-title">
            <span>📡 SISTEMA DE VISIÓN COMPUTACIONAL & TELEMETRÍA FOLIAR</span>
        </div>
        <div class="telecom-sub">Clasificador Digital de Patologías Foliares · Cultivos de Fresa</div>
        <div class="telecom-badge-bar">
            <span class="badge-chip">NODE: PAMPLONA-AGRO-01</span>
            <span class="badge-chip">DSP: HSV_COLOR_SPACE</span>
            <span class="badge-chip">FEATURE_ENGINE: GLCM_11D</span>
            <span class="badge-chip">AI_CORE: SCIKIT-KNN</span>
            <span class="badge-chip">UNAD / SIPROINT</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# 5. Carga de Pesos Serializados
@st.cache_resource
def cargar_archivos():
    modelo = joblib.load("modelo_knn_fresa.pkl")
    scaler = joblib.load("escalador_caracteristicas.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_archivos()
    st.sidebar.success("⚡ ENLACE DE PARÁMETROS ACTIVO (.PKL)")
except Exception as e:
    st.error(f"❌ FALLA EN ENLACE SERIALIZADO (.PKL): {e}")
    st.stop()

# 6. Pipeline DSP (Extracción de Características)
def segmentar_hoja(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lim_inf = np.array([15, 30, 30])
    lim_sup = np.array([95, 255, 255])
    mascara = cv2.inRange(hsv, lim_inf, lim_sup)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    return mascara

def extraer_vector(img_rgb):
    img_redim = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    mascara = segmentar_hoja(img_redim)
    pixeles = mascara > 0
    if np.sum(pixeles) < 1000:
        pixeles = np.ones((512, 512), dtype=bool)

    img_hsv = cv2.cvtColor(img_redim, cv2.COLOR_RGB2HSV)
    h_media, h_std = float(np.mean(img_hsv[:, :, 0][pixeles])), float(np.std(img_hsv[:, :, 0][pixeles]))
    s_media, s_std = float(np.mean(img_hsv[:, :, 1][pixeles])), float(np.std(img_hsv[:, :, 1][pixeles]))
    v_media, v_std = float(np.mean(img_hsv[:, :, 2][pixeles])), float(np.std(img_hsv[:, :, 2][pixeles]))

    img_gris = cv2.cvtColor(img_redim, cv2.COLOR_RGB2GRAY)
    cuantizada = (img_gris // 4).astype(np.uint8)
    glcm = graycomatrix(cuantizada, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64, symmetric=True, normed=True)

    contraste    = float(np.mean(graycoprops(glcm, 'contrast')))
    disimilitud  = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    homogeneidad = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energia      = float(np.mean(graycoprops(glcm, 'energy')))
    correlacion  = float(np.mean(graycoprops(glcm, 'correlation')))

    vector = [h_media, h_std, s_media, s_std, v_media, v_std, contraste, disimilitud, homogeneidad, energia, correlacion]
    nombres = ['H_MEAN', 'H_STD', 'S_MEAN', 'S_STD', 'V_MEAN', 'V_STD', 'GLCM_CONTRAST', 'GLCM_DISSIMILARITY', 'GLCM_HOMOGENEITY', 'GLCM_ENERGY', 'GLCM_CORRELATION']
    return vector, nombres

# 7. Ingesta de Datos del Sensor Óptico
st.markdown("#### 📥 INGESTA DE TRAMA ÓPTICA (SENSOR / CÁMARA)")
archivo_subido = st.file_uploader(
    "Cargar cuadro foliar RGB (Formato RAW/JPG/PNG):",
    type=["jpg", "jpeg", "png"]
)

if archivo_subido is not None:
    t_inicio = time.time()
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    img_np = np.array(imagen_pil)
    alto, ancho, _ = img_np.shape

    # Panel de Dos Columnas: Frame Óptico vs Telemetría
    col_frame, col_telemetria = st.columns([1, 1.3], gap="large")

    with col_frame:
        st.markdown("##### 🔲 BUFFER DEL FRAME CAPTURADO")
        st.image(imagen_pil, use_container_width=True)
        st.caption(f"Resolución nativa: {ancho}x{alto} px | Canales: RGB_8BIT | Interpolación: INTER_AREA")

    with st.spinner("Procesando señal digital y vectorizando descriptores..."):
        descriptores, nombres_desc = extraer_vector(img_np)
        vector_escalado = scaler.transform([descriptores])
        prediccion = modelo.predict(vector_escalado)[0]
        probabilidades = modelo.predict_proba(vector_escalado)[0]
        clases_modelo = list(modelo.classes_)
        latencia = (time.time() - t_inicio) * 1000

    with col_telemetria:
        # Colores de Veredicto
        es_sana = "sana" in prediccion.lower()
        color_neon = "#00f5d4" if es_sana else "#ff0054"
        icono_telecom = "🟢 OPTIMAL_HEALTH" if es_sana else "🔴 PATHOLOGY_ALERT"

        st.markdown(f"""
            <div class="verdict-box" style="border-left: 6px solid {color_neon};">
                <div class="verdict-header">TELEMETRY_CLASSIFICATION // TARGET_STATUS</div>
                <div class="verdict-title" style="color: {color_neon};">{prediccion.upper()}</div>
                <div style="font-size: 0.82rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                    SIGNAL: {icono_telecom} | LATENCY: {latencia:.1f} ms | CONF_METRIC: KNN_SOFTMAX
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📶 DENSIDAD ESPECTRAL DE PROBABILIDAD (CHANNELS 1-5)")

        todas_las_categorias = [
            ("Sana", "CH_01"),
            ("Viruela", "CH_02"),
            ("Antracnosis", "CH_03"),
            ("Oídio", "CH_04"),
            ("No reconocida / Dudosa", "CH_05")
        ]

        dict_modelo = {cls: prob * 100 for cls, prob in zip(clases_modelo, probabilidades)}

        filas_tabla = []
        for cat, ch in todas_las_categorias:
            prob_calculada = 0.0
            for k, v in dict_modelo.items():
                if cat.lower() in k.lower():
                    prob_calculada = v
                    break

            filas_tabla.append({
                "Canal": ch,
                "Patología": cat,
                "Probabilidad (%)": round(prob_calculada, 2)
            })

            # Barra de telemetría por canal
            col_ch1, col_ch2 = st.columns([3, 1])
            with col_ch1:
                st.write(f"`{ch}` **{cat}**")
                st.progress(int(min(max(prob_calculada, 0), 100)))
            with col_ch2:
                st.metric(label="", value=f"{prob_calculada:.1f}%")

    st.markdown("---")

    # Matriz DSP de 11 Descriptores
    with st.expander("🔬 TELEMETRÍA DSP: VECTOR DE CARACTERÍSTICAS (11D MULTIVARIABLE)"):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Hue Mean (H_μ)", f"{descriptores[0]:.2f}")
        col_m2.metric("Sat Mean (S_μ)", f"{descriptores[2]:.2f}")
        col_m3.metric("Val Mean (V_μ)", f"{descriptores[4]:.2f}")
        col_m4.metric("GLCM Contrast", f"{descriptores[6]:.2f}")

        df_telecom = pd.DataFrame({
            "Descriptor DSP": nombres_desc,
            "Valor Cuantizado": [round(val, 5) for val in descriptores],
            "Dominio": [
                "Colorimétrico HSV", "Colorimétrico HSV", "Colorimétrico HSV", "Colorimétrico HSV",
                "Colorimétrico HSV", "Colorimétrico HSV", "Espacial / Haralick", "Espacial / Haralick",
                "Espacial / Haralick", "Espacial / Haralick", "Espacial / Haralick"
            ]
        })
        st.dataframe(df_telecom, use_container_width=True)
