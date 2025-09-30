# app_uber_pred.py
# Streamlit app: Predicción de estado de reserva (Uber) basado en el modelo entrenado
# Uso: streamlit run app_uber_pred.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# 1) Cargar modelo y metadatos
# -------------------------------
@st.cache_resource
def load_model():
    try:
        modelTree,modelKnn, modelNN, modelSVM, labelencoder,variables,min_max_scaler  = pickle.load(open("modelo.pkl", "rb"))
    except Exception as e:
        st.error(f"No pude cargar 'Modelo_Regresion_UBER.pkl'. Asegúrate de tener el archivo en la misma carpeta. Error: {e}")
        st.stop()

    # 'variables' puede venir como ndarray -> lo convertimos a lista de strings
    if hasattr(variables, "tolist"):
        variables = [str(v) for v in variables.tolist()]
    else:
        variables = [str(v) for v in variables]

    return modelTree,modelKnn, modelNN, modelSVM, labelencoder,variables,min_max_scaler


model, variables, min_max_scaler, classes = load_model()

st.title("🚕 Predicción de 'Booking Status' (Uber)")
st.caption("Despliegue rápido del modelo entrenado con SVM")

def discretizar_hora(hora):
    if 5 <= hora < 12:
        return "mañana"
    elif 12 <= hora < 18:
        return "tarde"
    elif 18 <= hora < 24:
        return "noche"
    else:
        return "madrugada"

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
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
    data["dia_semana"] = data["fecha"].dt.dayofweek  # 0=lunes, 6=domingo

    # --- 3. Codificar hora de manera cíclica ---
    data["hora_sin"] = np.sin(2 * np.pi * data["hora"] / 24)
    data["hora_cos"] = np.cos(2 * np.pi * data["hora"] / 24)

    df = coerce_basic_types(df_raw)
    df = feature_engineering(df)
    df = stable_encode_categoricals(df)
    X = df.reindex(columns=variables, fill_value=0)
    return X


# -------------------------------
# 3) Entrada de datos
# -------------------------------
st.header("📥 Ingresar datos futuros")

mode = st.radio("¿Cómo quieres ingresar los datos?", ["📤 Subir CSV", "📝 Capturar 1 registro"], horizontal=True)

if mode == "📤 Subir CSV":
    file = st.file_uploader("Cargar CSV con el mismo esquema original (data.csv)", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df_raw = pd.read_csv(file, sep=";")
        st.subheader("Vista previa")
        st.dataframe(df_raw.head())

        if st.button("🔮 Predecir"):
            X = prepare_features(df_raw)
            try:
                y_pred = model.predict(X)
                proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X)
                    except Exception:
                        proba = None
                out = df_raw.copy()
                out["Prediccion"] = [classes[int(i)] for i in y_pred]
                if proba is not None:
                    for i, c in enumerate(classes):
                        out[f"proba_{c}"] = proba[:, i]
                st.success("Predicciones generadas.")
                st.dataframe(out.head())

                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar resultados (CSV)", data=csv, file_name="predicciones_uber.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Ocurrió un error prediciendo: {e}")

else:
    st.write("Captura un registro de ejemplo:")

    col1, col2, col3 = st.columns(3)
    with col1:
        ride_distance = st.number_input("Ride Distance (km)", min_value=0.0, value=12.0, step=0.5)
        booking_value = st.number_input("Booking Value", min_value=0.0, value=25.0, step=1.0)
    with col2:
        avg_ctat = st.number_input("Avg CTAT (min)", min_value=0.0, value=20.0, step=1.0)
        avg_vtat = st.number_input("Avg VTAT (min)", min_value=0.0, value=8.0, step=1.0)
    with col3:
        payment_method = st.selectbox("Payment Method", ["Cash", "CreditCard", "DebitCard", "UberWallet", "UPI"], index=1)

    df_single = pd.DataFrame([{
        "Ride Distance": ride_distance,
        "Booking Value": booking_value,
        "Avg CTAT": avg_ctat,
        "Avg VTAT": avg_vtat,
        "Payment Method": payment_method,
    }])

    if st.button("🔮 Predecir registro"):
        X = prepare_features(df_single)
        try:
            y_pred = model.predict(X)
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                except Exception:
                    proba = None
            st.success(f"Predicción: {classes[int(y_pred[0])]}")
            if proba is not None:
                st.write("Probabilidades por clase:")
                st.json({classes[i]: float(p) for i, p in enumerate(proba[0])})
        except Exception as e:
            st.error(f"Ocurrió un error prediciendo: {e}")
