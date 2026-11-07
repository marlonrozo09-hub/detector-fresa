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

# 2. Inyección de estilos CSS personalizados para color y diseño moderno
st.markdown("""
    <style>
    /* Estilo del encabezado principal */
    .header-box {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #c1121f 100%);
        padding: 24px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .header-box h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .header-box p {
        color: #e9ecef !important;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    /* Tarjetas de resultados */
    .diag-card {
        background: #f8fafc;
        border-left: 6px solid #2d6a4f;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Estilo del botón de carga */
    .stFileUploader label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1b4332;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Barra lateral estilizada con información del proyecto UNAD
with st.sidebar:
    st.markdown("### 🏛️ Semillero SIPROINT")
    st.caption("Universidad Nacional Abierta y a Distancia (UNAD)")
    st.markdown("---")
    st.markdown("**Proyecto:** Clasificador de hojas de fresa")
    st.markdown("**Zona:** Pamplona, Norte de Santander")
    st.markdown("**Modelo:** K-Nearest Neighbors (KNN)")
    st.markdown("**Pipeline:** Color HSV + Textura GLCM")
    st.markdown("---")
    st.subheader("Categorías Evaluadas")
    st.markdown("""
    - 🟢 **Sana**
    - 🟣 **Viruela** (*Ramularia tulasnei*)
    - 🟤 **Antracnosis** (*Colletotrichum spp.*)
    - ⚪ **Oídio** (*Podosphaera aphanis*)
    - ❓ **No reconocida / Dudosa**
    """)
    st.markdown("---")
    st.caption("Prototipo de clasificación foliar desarrollado con fines de investigación.")

# 4. Banner principal con color y degradado
st.markdown("""
    <div class="header-box">
        <h1>🍓 Clasificador de Imágenes de la Hoja de Fresa</h1>
        <p>Detección de viruela, antracnosis, oídio o no reconocidas mediante procesamiento de color, textura y algoritmo KNN</p>
        <p style="font-size: 0.88rem; opacity: 0.9; margin-top: 6px;">Cultivos a campo abierto · Pamplona, Norte de Santander</p>
    </div>
""", unsafe_allow_html=True)

# 5. Cargar modelos serializados (.pkl)
@st.cache_resource
def cargar_archivos():
    modelo = joblib.load("modelo_knn_fresa.pkl")
    scaler = joblib.load("escalador_caracteristicas.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_archivos()
    st.sidebar.success("Modelo KNN cargado y activo")
except Exception as e:
    st.error(f"Error al cargar los archivos .pkl: {e}")
    st.stop()

# 6. Pipeline de visión por computador
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

# 7. Carga de archivo
archivo_subido = st.file_uploader(
    "Seleccione o capture una fotografía de la hoja de fresa:",
    type=["jpg", "jpeg", "png"]
)

if archivo_subido is not None:
    col_img, col_diag = st.columns([1, 1.2], gap="large")

    imagen_pil = Image.open(archivo_subido).convert('RGB')
    img_np = np.array(imagen_pil)

    with col_img:
        st.subheader("Fotografía Ingresada")
        st.image(imagen_pil, use_container_width=True)

    with st.spinner("Analizando descriptores de color y textura con KNN..."):
        descriptores, nombres_desc = extraer_vector(img_np)
        vector_escalado = scaler.transform([descriptores])
        prediccion = modelo.predict(vector_escalado)[0]
        probabilidades = modelo.predict_proba(vector_escalado)[0]
        clases_modelo = list(modelo.classes_)

    with col_diag:
        # Colores e íconos dinámicos según el diagnóstico emitido
        color_borde = "#2d6a4f" if "sana" in prediccion.lower() else "#c1121f"
        icono = "🌿" if "sana" in prediccion.lower() else "🔍"

        st.markdown(f"""
            <div style="background: #ffffff; border-left: 6px solid {color_borde}; padding: 18px 22px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 20px;">
                <span style="font-size: 0.95rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Resultado de Clasificación</span>
                <h2 style="color: #0f172a; margin: 4px 0 0 0; font-size: 1.8rem;">{icono} {prediccion}</h2>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Nivel de concordancia por categoría:")

        todas_las_categorias = [
            ("Sana", "🟢"),
            ("Viruela", "🟣"),
            ("Antracnosis", "🟤"),
            ("Oídio", "⚪"),
            ("No reconocida / Dudosa", "❓")
        ]

        dict_modelo = {cls: prob * 100 for cls, prob in zip(clases_modelo, probabilidades)}

        filas_tabla = []
        for cat, emoji in todas_las_categorias:
            prob_calculada = 0.0
            for k, v in dict_modelo.items():
                if cat.lower() in k.lower():
                    prob_calculada = v
                    break

            filas_tabla.append({
                "Patología": f"{emoji} {cat}",
                "Certeza (%)": round(prob_calculada, 1)
            })

            # Barra visual de progreso por categoría
            st.write(f"**{emoji} {cat}:** `{prob_calculada:.1f}%`")
            st.progress(int(min(max(prob_calculada, 0), 100)))

    # Desplegable técnico con los 11 descriptores
    st.divider()
    with st.expander("📊 Ver vector de características extraído (11 descriptores numéricos)"):
        df_desc = pd.DataFrame({
            "Descriptor": nombres_desc,
            "Valor Numérico": [round(val, 4) for val in descriptores]
        })
        st.dataframe(df_desc, use_container_width=True)
