# entrenamiento_modelo.py

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# 1. Cargar y limpiar datasets
# -----------------------------
df_envios = pd.read_csv("tarifas_excel.csv")
df_tarifas = pd.read_csv("tarifas_pdf.csv")
df_xpo = pd.read_csv("tarifas_xpo.csv")
df_norte = pd.read_csv("tarifas_norte.csv")

# Renombrar columnas
df_envios = df_envios.rename(columns={
    'Punto de Origen': 'origen',
    'Punto Destino': 'destino',
    'Peso (kg)': 'peso_tasable',
    'Precio total (€)': 'precio_total',
    'Proveedor': 'proveedor'
})

df_tarifas = df_tarifas.rename(columns={
    'Punto de Origen': 'origen',
    'Punto Destino': 'destino',
    'Peso (kg)': 'peso_tasable',
    'Precio Total': 'precio_total',
    'Proveedor': 'proveedor'
})

df_xpo = df_xpo.rename(columns={
    'Punto de Origen': 'origen',
    'Punto Destino': 'destino',
    'peso': 'peso_tasable',
    'Precio Total': 'precio_total',
    'Proveedor': 'proveedor'
})

df_norte = df_norte.rename(columns={
    'Origen': 'origen',
    'Destino': 'destino',
    'Peso Tasable (kg)': 'peso_tasable',
    'Precio Total (€)': 'precio_total',
    'Proveedor': 'proveedor'
})

# Limpieza
for df in [df_envios, df_tarifas, df_xpo, df_norte]:
    df['peso_tasable'] = df['peso_tasable'].astype(str).str.replace(',', '.').astype(float)
    df['precio_total'] = df['precio_total'].astype(str).str.replace(',', '.').astype(float)

# Unión
df_completo = pd.concat([df_envios, df_tarifas, df_xpo, df_norte], ignore_index=True)

# 2. Simular proveedor base
rutas_pesos_reales = df_completo.groupby(['origen', 'destino'])['peso_tasable'].unique().reset_index()
filas_simuladas = []

for _, fila in rutas_pesos_reales.iterrows():
    origen, destino, pesos = fila['origen'], fila['destino'], fila['peso_tasable']
    for peso in pesos:
        precios = df_completo[(df_completo['origen'] == origen) & (df_completo['destino'] == destino) & (df_completo['peso_tasable'] == peso)]['precio_total']
        if not precios.empty:
            precio_real = precios.sample(1).values[0]
            precio_simulado = precio_real * np.random.uniform(0.9, 1.1)
        else:
            precio_simulado = 42 + 0.2 * peso

        filas_simuladas.append({
            'origen': origen,
            'destino': destino,
            'peso_tasable': peso,
            'precio_total': round(precio_simulado, 2),
            'proveedor': 'ProveedorBase'
        })

# Crear df_simulado y unir
df_simulado = pd.DataFrame(filas_simuladas)
df_completo = pd.concat([df_completo, df_simulado], ignore_index=True)

# 3. Agregar distancia
geolocator = Nominatim(user_agent="envio_modelo", timeout=10)
coordenadas_cache = {}

def limpiar_codigo(c): return c.replace('(', '').replace(')', '').strip()

def coordenadas_cacheadas(c):
    if c not in coordenadas_cache:
        try:
            lugar = limpiar_codigo(c)
            ubicacion = geolocator.geocode(lugar)
            coordenadas_cache[c] = (ubicacion.latitude, ubicacion.longitude) if ubicacion else None
        except:
            coordenadas_cache[c] = None
        time.sleep(1)
    return coordenadas_cache[c]

def calcular_distancia_km(fila):
    c1 = coordenadas_cacheadas(fila['origen'])
    c2 = coordenadas_cacheadas(fila['destino'])
    return geodesic(c1, c2).km if c1 and c2 else None

print("Calculando distancias...")
df_completo['distancia_km'] = df_completo.apply(calcular_distancia_km, axis=1)

# 4. Codificar y guardar
le_origen = LabelEncoder()
le_destino = LabelEncoder()
df_completo['origen_cod'] = le_origen.fit_transform(df_completo['origen'])
df_completo['destino_cod'] = le_destino.fit_transform(df_completo['destino'])

joblib.dump(le_origen, "le_origen.joblib")
joblib.dump(le_destino, "le_destino.joblib")

# Entrenamiento por proveedor
modelos = {}
param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

for proveedor in df_completo['proveedor'].unique():
    datos = df_completo[df_completo['proveedor'] == proveedor]
    X = datos[['origen_cod', 'destino_cod', 'peso_tasable', 'distancia_km']]
    y = datos['precio_total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    modelo_final = grid.best_estimator_
    modelos[proveedor] = modelo_final
    joblib.dump(modelo_final, f"modelo_{proveedor}.joblib")

# 5. EVALUACION
metricas = []
for proveedor, modelo in modelos.items():
    datos = df_completo[df_completo['proveedor'] == proveedor]
    X = datos[['origen_cod', 'destino_cod', 'peso_tasable', 'distancia_km']]
    y = datos['precio_total']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = modelo.predict(X_test)

    metricas.append({
        'proveedor': proveedor,
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'N_test': len(y_test)
    })

print("\n\n✨ Resultados de evaluación por proveedor:")
print(pd.DataFrame(metricas))


# 6. Guardar dataset completo
df_completo.to_csv("df_completo.csv", index=False, decimal=',')
print("\n✅ Entrenamiento y guardado completado.")
