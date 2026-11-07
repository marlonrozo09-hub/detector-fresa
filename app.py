import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# 1. Configuración de la pestaña del navegador
st.set_page_config(
    page_title="Clasificador KNN Hojas de Fresa - Pamplona",
    page_icon="🍓",
    layout="wide"
)

# 2. Barra lateral alineada al proyecto institucional UNAD
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

# 3. Título principal oficial del proyecto
st.title("🍓 Clasificador de Imágenes de la Hoja de Fresa")
st.subheader("Detección de viruela, antracnosis, oídio o imágenes no reconocidas mediante KNN")
st.caption("Prototipo para cultivos a campo abierto en el municipio de Pamplona, Norte de Santander.")

# 4. Carga de los archivos serializados
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

# 5. Pipeline de visión por computador (en segundo plano)
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

# 6. Interfaz de carga
archivo_subido = st.file_uploader(
    "Cargue o capture una fotografía de la hoja (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    img_np = np.array(imagen_pil)

    # Vista previa de la fotografía real
    st.subheader("Hoja Ingresada")
    st.image(imagen_pil, use_container_width=True)

    with st.spinner("Extrayendo descriptores y clasificando con KNN..."):
        descriptores, nombres_desc = extraer_vector(img_np)
        vector_escalado = scaler.transform([descriptores])
        prediccion = modelo.predict(vector_escalado)[0]
        probabilidades = modelo.predict_proba(vector_escalado)[0]
        clases_modelo = list(modelo.classes_)

    st.divider()

    # Diagnóstico Principal
    st.subheader(f"Diagnóstico Principal:
