import streamlit as st
import pandas as pd
import joblib
import os
from glob import glob
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ------------------ CONFIGURACION ------------------
st.set_page_config(page_title="Recomendador de Proveedor", layout="centered")

# LOGO alineado a la derecha usando columnas
col1, col2 = st.columns([6, 1])
with col2:
    st.image("zentralcom.png", width=300)

st.title("🚚 Recomendador de proveedor logístico")


# ------------------ CARGA DE DATOS Y MODELOS ------------------
st.info("Cargando modelos y datos...")
df = pd.read_csv("df_completo.csv", decimal=",")
modelos_por_proveedor = {os.path.basename(p).replace("modelo_", "").replace(".joblib", ""): joblib.load(p) for p in glob("modelo_*.joblib")}
le_origen = joblib.load("le_origen.joblib")
le_destino = joblib.load("le_destino.joblib")

geolocator = Nominatim(user_agent="envio_prediccion", timeout=10)
coordenadas_cache = {}

def limpiar_codigo(codigo):
    return codigo.replace('(', '').replace(')', '').strip()

def coordenadas_cacheadas(codigo):
    if codigo not in coordenadas_cache:
        try:
            lugar = limpiar_codigo(codigo)
            ubicacion = geolocator.geocode(lugar)
            coordenadas_cache[codigo] = (ubicacion.latitude, ubicacion.longitude) if ubicacion else None
        except:
            coordenadas_cache[codigo] = None
    return coordenadas_cache[codigo]

def calcular_distancia_km(origen, destino):
    coord_origen = coordenadas_cacheadas(origen)
    coord_destino = coordenadas_cacheadas(destino)
    if coord_origen and coord_destino:
        return geodesic(coord_origen, coord_destino).km
    return None

# ------------------ INPUTS DEL USUARIO ------------------
st.sidebar.header("Parámetros del envío")
distinct_origenes = sorted(df['origen'].unique())
distinct_destinos = sorted(df['destino'].unique())

origen_input = st.sidebar.selectbox("Origen", distinct_origenes)
destino_input = st.sidebar.selectbox("Destino", distinct_destinos)
peso_input = st.sidebar.number_input("Peso real (kg)", min_value=1, step=1)
volumen_input = st.sidebar.number_input("Volumen (m3)", min_value=0.0, step=0.1)
metros_lineales_input = st.sidebar.number_input("Metros lineales", min_value=0.0, step=0.1)
tarifa_actual = st.sidebar.number_input("Tarifa actual (€)", min_value=0.0, step=1.0)

if st.sidebar.button("Predecir proveedor más barato"):
    if volumen_input == 0.0 and metros_lineales_input == 0.0:
        st.error("Debes especificar al menos el volumen o los metros lineales.")
    else:
        origen_cod = le_origen.transform([origen_input])[0]
        destino_cod = le_destino.transform([destino_input])[0]
        distancia_km = calcular_distancia_km(origen_input, destino_input)

        factores_volumetricos = {'transnatur': 333, 'rhenus': 333, 'xpo': 333, 'default': 333}
        resultados = {}

        for proveedor, modelo in modelos_por_proveedor.items():
            proveedor_key = proveedor.strip().lower()
            factor = factores_volumetricos.get(proveedor_key, factores_volumetricos['default'])

            tiene_ruta = not df[(df['proveedor'] == proveedor) & (df['origen'] == origen_input) & (df['destino'] == destino_input)].empty
            if not tiene_ruta:
                continue

            if volumen_input > 0:
                peso_vol = volumen_input * factor
            elif metros_lineales_input > 0:
                peso_vol = metros_lineales_input * 1750
            else:
                peso_vol = 0

            peso_tasable = max(peso_input, peso_vol)

            X_nuevo = pd.DataFrame([{
                'origen_cod': origen_cod,
                'destino_cod': destino_cod,
                'peso_tasable': peso_tasable,
                'distancia_km': distancia_km
            }])

            precio_estimado = modelo.predict(X_nuevo)[0]
            resultados[proveedor] = precio_estimado

        if resultados:
            resultados_ordenados = sorted(resultados.items(), key=lambda x: x[1])
            mejor_proveedor, mejor_precio = resultados_ordenados[0]
            segundo_mejor = resultados_ordenados[1] if len(resultados_ordenados) > 1 else (None, None)

            st.success(f"Proveedor recomendado: **{mejor_proveedor}**")

            st.markdown(f"""
            ### 📦 Resumen del envío
            - Origen: **{origen_input}**
            - Destino: **{destino_input}**
            - Peso real: **{peso_input:.2f} kg**
            - Peso tasable usado: **{peso_tasable:.2f} kg**
            - Tarifa actual: **{tarifa_actual:.2f} €**
            """)




            st.write(f"💰 Precio estimado: **{mejor_precio:.2f} €**")

            if segundo_mejor[0]:
                diferencia = segundo_mejor[1] - mejor_precio
                diferencia_pct = (diferencia / segundo_mejor[1]) * 100 if segundo_mejor[1] else 0
                st.write(f"Segundo proveedor: **{segundo_mejor[0]}** → {segundo_mejor[1]:.2f} €")
                st.write(f"Diferencia: **{diferencia:.2f} € ({diferencia_pct:.1f} %)**")

            ahorro = tarifa_actual - mejor_precio
            ahorro_pct = (ahorro / tarifa_actual) * 100 if tarifa_actual else 0
            if ahorro > 0:
                st.success(f"Ahorro estimado respecto a la tarifa actual: **{ahorro:.2f} € ({ahorro_pct:.1f} %)**")
            elif ahorro < 0:
                st.warning(f"La tarifa actual es más baja por **{-ahorro:.2f} € ({-ahorro_pct:.1f} %)**")
            else:
                st.info("La tarifa actual coincide con la predicción del modelo.")
        else:
            st.error("Ningún proveedor tiene datos suficientes para predecir esta ruta.")
