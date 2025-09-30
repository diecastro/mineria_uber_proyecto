#%% md
# #Despliegue
# 
# - Cargamos el modelo
# 
# - Cargamos los datos futuros
# 
# - Preparar los datos futuros
# 
# - Aplicamos el modelo para la predicción
#%%
#importamos librerias básicas
import pandas as pd #manipulación dataframes
import numpy as np #matrices y vectores
import pickle
import streamlit as st
#%%
#Cargamos el modelo
filename = 'modelo.pkl'
modelTree,modelKnn, modelNN, modelSVM, labelencoder,variables,min_max_scaler = pickle.load(open(filename, 'rb'))
#%%
st.header("📥 Ingresar datos futuros")

mode = st.radio("¿Cómo quieres ingresar los datos?", ["📤 Subir CSV"], horizontal=True)

file = st.file_uploader("Cargar CSV con el mismo esquema original (data.csv)", type=["csv"])


#Cargamos los datos futuros
data=pd.read_excel(file)

st.subheader("Vista previa")
st.dataframe(df_raw.head())

#%%
# Re-create the 'DateTime' column by combining 'Date' and 'Time'
data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), errors='coerce')
# Drop the original 'Date' and 'Time' columns
data = data.drop(['Date', 'Time'], axis=1)
# --- 1. Convertir DateTime ---
# The 'DateTime' column already exists and is in datetime format
data["fecha"] = data["DateTime"]
data["hora"] = data["DateTime"].dt.hour

# --- 2. Extraer características de fecha ---
data["anio"] = data["fecha"].dt.year
data["mes"] = data["fecha"].dt.month
data["dia_semana"] = data["fecha"].dt.dayofweek   # 0=lunes, 6=domingo

# --- 3. Codificar hora de manera cíclica ---
data["hora_sin"] = np.sin(2 * np.pi * data["hora"] / 24)
data["hora_cos"] = np.cos(2 * np.pi * data["hora"] / 24)

# --- 4. (Opcional) Discretizar hora en franjas horarias ---
def discretizar_hora(hora):
    if 5 <= hora < 12:
        return "mañana"
    elif 12 <= hora < 18:
        return "tarde"
    elif 18 <= hora < 24:
        return "noche"
    else:
        return "madrugada"

data["franja_horaria"] = data["hora"].apply(discretizar_hora)

# Convertir franja_horaria en variables dummy
data = pd.get_dummies(data, columns=["franja_horaria"], dtype=int)
# Función para asignar estación según mes
def asignar_estacion(mes):
    if mes in [12, 1, 2]:
        return "Invierno"
    elif mes in [3, 4, 5]:
        return "Verano"
    elif mes in [6, 7, 8, 9]:
        return "Monzon"
    else:  # 10, 11
        return "Post-Monzon"

# Crear columna de estación
data["estacion"] = data["mes"].map(asignar_estacion)
# Crear variables dummies sin borrar nada más
data = pd.get_dummies(data, columns=["estacion"], drop_first=False, dtype=int)
data = data.drop(['DateTime','fecha','hora','anio','mes','dia_semana','hora_cos','hora_sin'], axis=1)

#%%
#Se realiza la preparación
data_preparada=data.copy()
data_preparada = pd.get_dummies(data_preparada, columns=['Vehicle Type','Booking Status','Cancelled Rides by Customer','Reason for cancelling by Customer','Cancelled Rides by Driver','Driver Cancellation Reason','Incomplete Rides','Incomplete Rides Reason','Payment Method'], drop_first=False) #En despliegue no se borran dummies

#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
data_preparada.head()
#%%
# Y_fut= modelTree.predict(data_preparada)
# print(labelencoder.inverse_transform(Y_fut))
#%%
#Hacemos la predicción con Knn
# Y_fut = modelKnn.predict(data_preparada)
# data['Prediccion Knn']=labelencoder.inverse_transform(Y_fut)
# print(data['Prediccion Knn'])
#%%
#Hacemos la predicción con Red Neuronal
# Y_fut = modelNN.predict(data_preparada)
# data['Prediccion Neural_Network']=labelencoder.inverse_transform(Y_fut)
# print(data['Prediccion Neural_Network'])
#%%

# Hacemos la predicción con Máquina soporte vectorial
Y_fut = modelSVM.predict(data_preparada)
st.success("Predicciones generadas.")
st.dataframe(data['Prediccion SVM'])

# data['Prediccion SVM']=labelencoder.inverse_transform(Y_fut)
# print(data['Prediccion SVM'])