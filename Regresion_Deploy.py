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
        model_gbc, variables, min_max_scaler, classes = pickle.load(open("Modelo_Regresion_UBER.pkl", "rb"))
    except Exception as e:
        st.error(f"No pude cargar 'Modelo_Regresion_UBER.pkl'. Asegúrate de tener el archivo en la misma carpeta. Error: {e}")
        st.stop()

    if hasattr(variables, "tolist"):
        variables = [str(v) for v in variables.tolist()]
    else:
        variables = [str(v) for v in variables]

    if hasattr(classes, "tolist"):
        classes = classes.tolist()

    return model_gbc, variables, min_max_scaler, classes


model, variables, min_max_scaler, classes = load_model()

st.title("🚕 Predicción de 'Booking Status' (Uber) Regresión")
st.caption("Despliegue rápido del modelo entrenado con GradientBoostingClassifier")

PAGO_MAP = {
    "DebitCard": "Debit",
    "CreditCard": "Debit",
    "UberWallet": "Debit",
    "Cash": "Cash",
    "UPI": "Debit",
}

CATS = {
    "Payment Method": ["Cash", "Debit"],
    "Ride Distance": ["Corta", "Intermedia", "Extensa"],
    "Avg CTAT": ["Alto", "Medio", "Bajo"],
    "Avg VTAT": ["Bajo", "Medio", "Alto"],
}

DROP_ALWAYS = [
    "Booking ID", "Customer ID", "Vehicle Type", "Pickup Location",
    "Drop Location", "Date", "Time", "Customer Rating", "Driver Ratings",
]

DROP_LEAKAGE = [
    "Driver Cancellation Reason",
    "Cancelled Rides by Driver",
    "Reason for cancelling by Customer",
    "Cancelled Rides by Customer",
    "Incomplete Rides Reason",
    "Incomplete Rides",
]

DISTANCE_BINS = [0, 15, 30, float("inf")]
DISTANCE_LABELS = ["Corta", "Intermedia", "Extensa"]

CTAT_BINS = [0, 10, 45, float("inf")]
CTAT_LABELS = ["Alto", "Medio", "Bajo"]

VTAT_BINS = [2, 7, 12, 20]
VTAT_LABELS = ["Bajo", "Medio", "Alto"]


def coerce_basic_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Vehicle Type", "Booking Status", "Reason for cancelling by Customer",
                "Driver Cancellation Reason", "Incomplete Rides Reason", "Payment Method"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    for col in ["Pickup Location", "Drop Location", "Booking ID", "Customer ID"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.time
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Payment Method" in df.columns:
        df["Payment Method"] = df["Payment Method"].map(PAGO_MAP).fillna("Debit")
    else:
        df["Payment Method"] = "Debit"

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

    for col in DROP_ALWAYS + DROP_LEAKAGE:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def stable_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, cats in CATS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col].astype(str), categories=cats, ordered=True)
            df[col] = df[col].cat.codes
    for col in df.select_dtypes(include=["object", "string"]).columns:
        uniq = sorted(df[col].astype(str).fillna("NA").unique().tolist())
        df[col] = pd.Categorical(df[col].astype(str).fillna("NA"), categories=uniq, ordered=True).codes
    return df


def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = coerce_basic_types(df_raw)
    df = feature_engineering(df)
    df = stable_encode_categoricals(df)
    X = df.reindex(columns=variables, fill_value=0)
    return X


st.header("📥 Ingresar datos futuros")

mode = st.radio("¿Cómo quieres ingresar los datos?", ["📤 Subir CSV", "📝 Capturar 1 registro"], horizontal=True)

with open("./data/BDUBERDatosFuturos.csv", "rb") as f:
    st.download_button(
        label="Descargar Plantilla De Ejemplo",
        data=f,
        file_name="plantilla_regresion.csv",
        mime="text/csv"
    )
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
