
# app_uber_pred.py
# Streamlit app: Predicción de estado de reserva (Uber) basado en el modelo entrenado
# Uso: streamlit run app_uber_pred.py

import io
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
        model_gbc, variables, min_max_scaler = pickle.load(open("Modelo_Regresion_UBER.pkl", "rb"))
    except Exception as e:
        st.error(f"No pude cargar 'Modelo_Regresion_UBER.pkl'. Asegúrate de tener el archivo en la misma carpeta. Error: {e}")
        st.stop()
    # 'variables' puede venir como ndarray -> lo convertimos a lista de strings
    if hasattr(variables, "tolist"):
        variables = [str(v) for v in variables.tolist()]
    else:
        variables = [str(v) for v in variables]
    return model_gbc, variables, min_max_scaler

model, variables, min_max_scaler = load_model()

st.title("🚕 Predicción de 'Booking Status' (Uber)")
st.caption("Despliegue rápido del modelo entrenado con GradientBoostingClassifier (70/30).")

# -------------------------------
# 2) Explicar pipeline de preparación (idéntico al notebook)
# -------------------------------
with st.expander("Ver pipeline de preparación aplicado (resumen)"):
    st.markdown(
        '''
        **Pasos reproducidos del notebook:**
        1. Corrección de tipos básicos (Date, Time, strings, categorías).
        2. **Selección** y **reducción**: se eliminan columnas irrelevantes y de fuga de información.
        3. **Reducción de dimensionalidad** de negocio:
           - `Payment Method` se mapea a `Cash` o `Debit`.
           - `Ride Distance` se discretiza en `Corta`, `Intermedia`, `Extensa` con *bins* `[0, 15, 30, inf)`.
           - `Avg CTAT` se discretiza en `Alto`, `Medio`, `Bajo` con *bins* `[0, 10, 45, inf)`.
           - `Avg VTAT` se discretiza en `Bajo`, `Medio`, `Alto` con *bins* `[2, 7, 12, 20]`.
        4. **Codificación**: variables categóricas se convierten a códigos numéricos *determinísticos*.
        5. Reindex de columnas para que coincidan con el orden de `variables` guardadas durante el entrenamiento.
        '''
    )

# -------------------------------
# 3) Funciones de preparación
# -------------------------------

PAGO_MAP = {
    "DebitCard": "Debit",
    "CreditCard": "Debit",
    "UberWallet": "Debit",
    "Cash": "Cash",
    "UPI": "Debit",
}

# Categorías fijas y ordenadas para codificación estable
CATS = {
    "Payment Method": ["Cash", "Debit"],
    "Ride Distance": ["Corta", "Intermedia", "Extensa"],
    "Avg CTAT": ["Alto", "Medio", "Bajo"],
    "Avg VTAT": ["Bajo", "Medio", "Alto"],
}

# Columnas descartadas en el notebook
DROP_ALWAYS = [
    "Booking ID", "Customer ID", "Vehicle Type", "Pickup Location",
    "Drop Location", "Date", "Time", "Customer Rating", "Driver Ratings",
]

# Fuga de información respecto al target
DROP_LEAKAGE = [
    "Driver Cancellation Reason",
    "Cancelled Rides by Driver",
    "Reason for cancelling by Customer",
    "Cancelled Rides by Customer",
    "Incomplete Rides Reason",
    "Incomplete Rides",
]

# Bins/labels usados en notebook
DISTANCE_BINS = [0, 15, 30, float("inf")]
DISTANCE_LABELS = ["Corta", "Intermedia", "Extensa"]

CTAT_BINS = [0, 10, 45, float("inf")]
CTAT_LABELS = ["Alto", "Medio", "Bajo"]

VTAT_BINS = [2, 7, 12, 20]
VTAT_LABELS = ["Bajo", "Medio", "Alto"]

def coerce_basic_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Tipos básicos usados en el notebook original
    for col in ["Vehicle Type", "Booking Status", "Reason for cancelling by Customer",
                "Driver Cancellation Reason", "Incomplete Rides Reason", "Payment Method"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    for col in ["Pickup Location", "Drop Location", "Booking ID", "Customer ID"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    # Date/Time pueden venir como texto
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if "Time" in df.columns:
        # Si trae HH:MM:SS perfecto; si no, intentamos parseo flexible
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.time
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Mapear Payment Method -> {Cash, Debit}
    if "Payment Method" in df.columns:
        df["Payment Method"] = df["Payment Method"].map(PAGO_MAP).fillna("Debit")
    else:
        df["Payment Method"] = "Debit"

    # Discretizaciones (igual que el notebook)
    if "Ride Distance" in df.columns:
        df["Ride Distance"] = pd.to_numeric(df["Ride Distance"], errors="coerce")
        df["Ride Distance"] = pd.cut(df["Ride Distance"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, include_lowest=True)
    else:
        df["Ride Distance"] = pd.Categorical(["Intermedia"] * len(df), categories=DISTANCE_LABELS)

    if "Avg CTAT" in df.columns:
        df["Avg CTAT"] = pd.to_numeric(df["Avg CTAT"], errors="coerce")
        df["Avg CTAT"] = pd.cut(df["Avg CTAT"], bins=CTAT_BINS, labels=CTAT_LABELS, include_lowest=True)
    else:
        df["Avg CTAT"] = pd.Categorical(["Medio"] * len(df), categories=CTAT_LABELS)

    if "Avg VTAT" in df.columns:
        df["Avg VTAT"] = pd.to_numeric(df["Avg VTAT"], errors="coerce")
        df["Avg VTAT"] = pd.cut(df["Avg VTAT"], bins=VTAT_BINS, labels=VTAT_LABELS, include_lowest=True)
    else:
        df["Avg VTAT"] = pd.Categorical(["Medio"] * len(df), categories=VTAT_LABELS)

    # Eliminar columnas irrelevantes y de fuga
    for col in DROP_ALWAYS + DROP_LEAKAGE:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

def stable_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Forzar categorías con orden prefijado
    for col, cats in CATS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col].astype(str), categories=cats, ordered=True)
            df[col] = df[col].cat.codes  # 0..n-1 (determinístico)
    # Convertir cualquier otro object/string a códigos con orden alfabético (determinístico)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        uniq = sorted(df[col].astype(str).fillna("NA").unique().tolist())
        df[col] = pd.Categorical(df[col].astype(str).fillna("NA"), categories=uniq, ordered=True).codes
    return df

def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = coerce_basic_types(df_raw)
    df = feature_engineering(df)
    df = stable_encode_categoricals(df)
    # Alinear columnas esperadas por el modelo
    X = df.reindex(columns=variables, fill_value=0)
    return X

# -------------------------------
# 4) Entrada de datos
# -------------------------------
st.header("📥 Ingresar datos futuros")

mode = st.radio("¿Cómo quieres ingresar los datos?", ["📤 Subir CSV", "📝 Capturar 1 registro"], horizontal=True)

if mode == "📤 Subir CSV":
    file = st.file_uploader("Cargar CSV con el mismo esquema original (data.csv)", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
        except Exception:
            # Intento extra con ; como separador
            file.seek(0)
            df_raw = pd.read_csv(file, sep=";")
        st.subheader("Vista previa")
        st.dataframe(df_raw.head())

        if st.button("🔮 Predecir"):
            X = prepare_features(df_raw)
            try:
                y_pred = model.predict(X)
                # Probabilidades (si existen)
                proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X)
                    except Exception:
                        proba = None
                out = df_raw.copy()
                out["Prediccion"] = y_pred
                if proba is not None:
                    # Agregar columnas proba_{i}
                    for i in range(proba.shape[1]):
                        out[f"proba_{i}"] = proba[:, i]
                st.success("Predicciones generadas.")
                st.dataframe(out.head())

                # Descargar resultado
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar resultados (CSV)", data=csv, file_name="predicciones_uber.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Ocurrió un error prediciendo: {e}")

else:
    # Captura de un registro mínimo viable
    st.write("Captura un registro de ejemplo (usa el esquema original, lo necesario para las transformaciones).")

    col1, col2, col3 = st.columns(3)
    with col1:
        ride_distance = st.number_input("Ride Distance (km)", min_value=0.0, value=12.0, step=0.5)
        booking_value = st.number_input("Booking Value", min_value=0.0, value=25.0, step=1.0)
    with col2:
        avg_ctat = st.number_input("Avg CTAT (min)", min_value=0.0, value=20.0, step=1.0)
        avg_vtat = st.number_input("Avg VTAT (min)", min_value=0.0, value=8.0, step=1.0)
    with col3:
        payment_method = st.selectbox("Payment Method", ["Cash", "CreditCard", "DebitCard", "UberWallet", "UPI"], index=1)

    # Construimos un DataFrame similar al original (mínimas columnas necesarias)
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
            st.success(f"Predicción: {int(y_pred[0])}")
            if proba is not None:
                st.write("Probabilidades por clase (índice):")
                st.json({str(i): float(p) for i, p in enumerate(proba[0])})
        except Exception as e:
            st.error(f"Ocurrió un error prediciendo: {e}")
