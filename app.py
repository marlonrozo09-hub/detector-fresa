import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# Configuración visual de la página
st.set_page_config(page_title="Diagnóstico Foliar de Fresa", layout="centered")

st.title("🍓 Sistema de Diagnóstico Fitosanitario de Fresa")
st.write("Carga una fotografía de una hoja para analizarla con visión por computador y KNN.")

# Cargar el modelo y escalador entrenados
@st.cache_resource
def cargar_archivos():
    modelo = joblib.load("modelo_knn_fresa.pkl")
    scaler = joblib.load("escalador_caracteristicas.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_archivos()
    st.success("Modelo y calibrador cargados correctamente.")
except Exception as e:
    st.error(f"Error cargando los archivos .pkl: {e}")
    st.stop()

# Funciones de extracción matemática
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

    return [h_media, h_std, s_media, s_std, v_media, v_std, contraste, disimilitud, homogeneidad, energia, correlacion]

# Subir la imagen
archivo_subido = st.file_uploader("Selecciona una imagen foliar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    st.image(imagen_pil, caption="Hoja ingresada", use_column_width=True)

    img_np = np.array(imagen_pil)
    with st.spinner("Procesando imagen y calculando descriptores..."):
        descriptores = extraer_vector(img_np)
        vector_escalado = scaler.transform([descriptores])
        prediccion = modelo.predict(vector_escalado)[0]
        probabilidades = modelo.predict_proba(vector_escalado)[0]
        clases = modelo.classes_

    st.subheader(f"Diagnóstico emitido: **{prediccion}**")

    # Mostrar nivel de certeza por clase
    st.write("### Nivel de concordancia por patología:")
    for cls, prob in zip(clases, probabilidades):
        st.write(f"- **{cls}:** {prob * 100:.1f}%")
