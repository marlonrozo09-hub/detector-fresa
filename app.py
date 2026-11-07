import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# 1. Configuración de página
st.set_page_config(
    page_title="Clasificador KNN Hojas de Fresa - Pamplona",
    page_icon="🍓",
    layout="wide"
)

# 2. Inyección de estilos CSS: Color oscuro tecnológico y acentos neón
st.markdown("""
    <style>
    /* Estilos generales y fondo del banner principal */
    .header-box {
        background: linear-gradient(135deg, #09111e 0%, #102a43 60%, #1e3a5f 100%);
        border: 1px solid #1f3a5f;
        border-left: 6px solid #00f5d4;
        padding: 24px 28px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 245, 212, 0.12);
    }
    .header-box h1 {
        color: #00f5d4 !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .header-box h3 {
        color: #e2e8f0 !important;
        font-size: 1.15rem;
        font-weight: 500;
        margin-top: 0px;
        margin-bottom: 6px;
    }
    .header-box p {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        margin-bottom: 0px;
    }

    /* Caja de diagnóstico principal */
    .resultado-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 6px solid #00f5d4;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .resultado-label {
        font-size: 0.85rem;
        color: #00b4d8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .resultado-valor {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Filas de patologías */
    .fila-patologia {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Barra lateral con tu información intacta
with st.sidebar:
    st.title("Semillero SIPROINT - UNAD")
    st.markdown("**Proyecto de Grado:**")
    st.markdown("Clasificador de imágenes foliares en fresa")
    st.markdown("**Ubicación:** Pamplona, Norte de Santander")
    st.markdown("**Algoritmo:** K-Nearest Neighbors (KNN)")
    st.markdown("**Descriptores:** Color (HSV) + Textura (GLCM)")
    st.divider()
    st.subheader("Categorías de Referencia")
    st.markdown("""
    - 🟢 **Sana**
    - 🟣 **Viruela** (*Ramularia tulasnei*)
    - 🟤 **Antracnosis** (*Colletotrichum spp.*)
    - ⚪ **Oídio** (*Podosphaera aphanis*)
    - ❓ **No reconocida / Dudosa**
    """)
    st.divider()
    st.caption("Prototipo de clasificación foliar desarrollado con fines de investigación.")

# 4. Banner superior con tus títulos oficiales y el nuevo estilo de color
st.markdown("""
    <div class="header-box">
        <h1>🍓 Clasificador de Imágenes de la Hoja de Fresa</h1>
        <h3>Detección de viruela, antracnosis, oídio o imágenes no reconocidas mediante KNN</h3>
        <p>Prototipo para cultivos a campo abierto en el municipio de Pamplona, Norte de Santander.</p>
    </div>
""", unsafe_allow_html=True)

# 5. Carga de los archivos serializados
@st.cache_resource
def cargar_archivos():
    modelo = joblib.load("modelo_knn_fresa.pkl")
    scaler = joblib.load("escalador_caracteristicas.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_archivos()
    st.sidebar.success("Modelo KNN activo y conectado")
except Exception as e:
    st.error(f"Error al cargar los archivos .pkl: {e}")
    st.stop()

# 6. Pipeline de procesamiento
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
    nombres = ['H Media', 'H Desv', 'S Media', 'S Desv', 'V Media', 'V Desv', 'Contraste', 'Disimilitud', 'Homogeneidad', 'Energía', 'Correlación']
    return vector, nombres

# 7. Interfaz de carga
archivo_subido = st.file_uploader(
    "Cargue o capture una fotografía de la hoja (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    img_np = np.array(imagen_pil)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("Hoja Ingresada")
        st.image(imagen_pil, use_container_width=True)

    with st.spinner("Extrayendo descriptores y clasificando con KNN..."):
        descriptores, nombres_desc = extraer_vector(img_np)
        vector_escalado = scaler.transform([descriptores])
        prediccion = modelo.predict(vector_escalado)[0]
        probabilidades = modelo.predict_proba(vector_escalado)[0]
        clases_modelo = list(modelo.classes_)

    with col2:
        # Diagnóstico principal en tarjeta con borde neón
        color_borde = "#00f5d4" if "sana" in prediccion.lower() else "#ff0054"
        st.markdown(f"""
            <div class="resultado-card" style="border-left-color: {color_borde};">
                <div class="resultado-label">Diagnóstico Principal Emitido</div>
                <div class="resultado-valor" style="color: {color_borde};">{prediccion}</div>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("Nivel de concordancia por categoría:")

        # Las 5 categorías oficiales completas
        todas_las_categorias = [
            ("Sana", "🟢"),
            ("Viruela", "🟣"),
            ("Antracnosis", "🟤"),
            ("Oídio", "⚪"),
            ("No reconocida / Dudosa", "❓")
        ]

        dict_modelo = {cls: prob * 100 for cls, prob in zip(clases_modelo, probabilidades)}

        for cat, emoji in todas_las_categorias:
            prob_calculada = 0.0
            for k, v in dict_modelo.items():
                if cat.lower() in k.lower():
                    prob_calculada = v
                    break

            col_texto, col_valor = st.columns([3, 1])
            with col_texto:
                st.write(f"**{emoji} {cat}**")
                st.progress(int(min(max(prob_calculada, 0), 100)))
            with col_valor:
                st.write(f"`{prob_calculada:.1f}%`")

    st.divider()

    # Detalle de los 11 descriptores
    with st.expander("Ver vector de características extraído (11 descriptores)"):
        df_desc = pd.DataFrame({
            "Descriptor": nombres_desc,
            "Valor Obtenido": [round(val, 4) for val in descriptores]
        })
        st.dataframe(df_desc, use_container_width=True)
