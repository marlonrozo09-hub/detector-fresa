import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# 1. Configuración de página
st.set_page_config(
    page_title="Diagnóstico Foliar de Fresa - Pamplona",
    page_icon="🍓",
    layout="wide"
)

# 2. Barra lateral con información del proyecto UNAD
with st.sidebar:
    st.title("Proyecto Fitosanitario")
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

# 3. Título principal
st.title("🍓 Sistema de Diagnóstico Fitosanitario en Hojas de Fresa")
st.write(
    "Herramienta computacional para la clasificación de patologías foliares "
    "mediante procesamiento digital de imágenes y modelo KNN."
)

# 4. Cargar modelo serializado
@st.cache_resource
def cargar_archivos():
    modelo = joblib.load("modelo_knn_fresa.pkl")
    scaler = joblib.load("escalador_caracteristicas.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_archivos()
    st.sidebar.success("Modelo KNN cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar los archivos .pkl: {e}")
    st.stop()

# 5. Pipeline de visión por computador
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
    return vector, nombres, mascara

# 6. Módulo de carga de imagen
archivo_subido = st.file_uploader(
    "Cargue o capture una imagen foliar (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    img_np = np.array(imagen_pil)

    with st.spinner("Extrayendo descriptores y clasificando..."):
        descriptores, nombres_desc, mascara = extraer_vector(img_np)
        vector_escalado = scaler.transform([descriptores])
        prediccion = modelo.predict(vector_escalado)[0]
        probabilidades = modelo.predict_proba(vector_escalado)[0]
        clases_modelo = list(modelo.classes_)

    # Visualización en columnas
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hoja Ingresada")
        st.image(imagen_pil, use_container_width=True)
    with col2:
        st.subheader("Segmentación Folio-Fondo")
        st.image(mascara, caption="Máscara binaria HSV", use_container_width=True)

    st.divider()

    # Diagnóstico Principal
    st.subheader(f"Diagnóstico Principal: **{prediccion}**")

    # 7. LAS 5 CATEGORÍAS COMPLETAS OBLIGATORIAS
    todas_las_categorias = [
        "Sana",
        "Viruela",
        "Antracnosis",
        "Oídio",
        "No reconocida / Dudosa"
    ]

    # Diccionario con lo que calculó el modelo
    dict_modelo = {cls: prob * 100 for cls, prob in zip(clases_modelo, probabilidades)}

    st.write("### Nivel de concordancia por patología:")
    
    # Construcción de la tabla completa fija
    filas_tabla = []
    for cat in todas_las_categorias:
        # Busca si la clase está en el modelo (o una versión que contenga el nombre)
        prob_calculada = 0.0
        for k, v in dict_modelo.items():
            if cat.lower() in k.lower():
                prob_calculada = v
                break

        filas_tabla.append({
            "Patología": cat,
            "Certeza (%)": round(prob_calculada, 1)
        })
        
        # Muestra la viñeta con el formato exacto que buscas
        st.markdown(f"* **{cat}:** {prob_calculada:.1f}%")

    # Gráfico de barras horizontal con todas las categorías
    df_grafico = pd.DataFrame(filas_tabla).set_index("Patología")
    st.bar_chart(df_grafico)

    # 8. Desplegable técnico con los 11 descriptores
    with st.expander("Ver vector de características extraído (11 descriptores)"):
        df_desc = pd.DataFrame({
            "Descriptor": nombres_desc,
            "Valor Obtenido": [round(val, 4) for val in descriptores]
        })
        st.dataframe(df_desc, use_container_width=True)
