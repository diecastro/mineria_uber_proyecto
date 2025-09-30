import pickle

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    try:
        modelTree, modelKnn, modelNN, modelSVM, labelencoder, variables, min_max_scaler = pickle.load(
            open("modelo.pkl", "rb"))
    except Exception as e:
        st.error(
            f"No pude cargar 'modelo.pkl'. Asegúrate de tener el archivo en la misma carpeta. Error: {e}")
        st.stop()

    if hasattr(variables, "tolist"):
        variables = [str(v) for v in variables.tolist()]
    else:
        variables = [str(v) for v in variables]

    return modelTree, modelKnn, modelNN, modelSVM, labelencoder, variables, min_max_scaler


modelTree, modelKnn, modelNN, modelSVM, labelencoder, variables, min_max_scaler = load_model()

st.title("🚕 Predicción de 'Booking Status' (Uber) Clasificación")
st.caption("Despliegue del modelo entrenado con SVM")


def discretizar_hora(hora):
    if 5 <= hora < 12:
        return "mañana"
    elif 12 <= hora < 18:
        return "tarde"
    elif 18 <= hora < 24:
        return "noche"
    else:
        return "madrugada"


def asignar_estacion(mes):
    if mes in [12, 1, 2]:
        return "Invierno"
    elif mes in [3, 4, 5]:
        return "Verano"
    elif mes in [6, 7, 8, 9]:
        return "Monzon"
    else:  # 10, 11
        return "Post-Monzon"


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
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
    data["dia_semana"] = data["fecha"].dt.dayofweek  # 0=lunes, 6=domingo

    # --- 3. Codificar hora de manera cíclica ---
    data["hora_sin"] = np.sin(2 * np.pi * data["hora"] / 24)
    data["hora_cos"] = np.cos(2 * np.pi * data["hora"] / 24)
    data["franja_horaria"] = data["hora"].apply(discretizar_hora)
    data = pd.get_dummies(data, columns=["franja_horaria"], dtype=int)
    data["estacion"] = data["mes"].map(asignar_estacion)
    # Crear variables dummies sin borrar nada más
    data = pd.get_dummies(data, columns=["estacion"], drop_first=False, dtype=int)
    data = data.drop(['DateTime', 'fecha', 'hora', 'anio', 'mes', 'dia_semana', 'hora_cos', 'hora_sin'], axis=1)
    data_preparada = data.copy()
    data_preparada = pd.get_dummies(data_preparada,
                                    columns=['Vehicle Type', 'Booking Status', 'Cancelled Rides by Customer',
                                             'Reason for cancelling by Customer', 'Cancelled Rides by Driver',
                                             'Driver Cancellation Reason', 'Incomplete Rides',
                                             'Incomplete Rides Reason', 'Payment Method'],
                                    drop_first=False)  # En despliegue no se borran dummies

    # Se adicionan las columnas faltantes
    X = data_preparada.reindex(columns=variables, fill_value=0)
    return X

st.header("📥 Ingresar datos futuros")

mode = st.radio("¿Cómo quieres ingresar los datos?", ["📤 Subir CSV", "📝 Capturar 1 registro"], horizontal=True)

if mode == "📤 Subir CSV":
    file = st.file_uploader("Cargar Archivo con el mismo esquema original", type=["csv", "xlsx","xls"])
    if file is not None:
        try:
            df_raw = pd.read_excel(file)
        except Exception:
            file.seek(0)
            try:
                df_raw = pd.read_csv(file, sep=";", encoding="utf-8")
            except UnicodeDecodeError:
                file.seek(0)
                df_raw = pd.read_csv(file, sep=";", encoding="latin-1")
            except Exception:
                file.seek(0)
                df_raw = pd.read_csv(file, sep=None, engine="python", encoding="latin-1")
        st.subheader("Vista previa")
        st.dataframe(df_raw.head())

        if st.button("🔮 Predecir"):
            X = prepare_features(df_raw)
            try:
                y_pred = modelSVM.predict(X)
                proba = None
                if hasattr(modelSVM, "predict_proba"):
                    try:
                        proba = modelSVM.predict_proba(X)
                    except Exception:
                        proba = None
                out = df_raw.copy()
                out['Prediccion SVM'] = labelencoder.inverse_transform(y_pred)
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar resultados (CSV)", data=csv, file_name="predicciones_uber.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(f"Ocurrió un error prediciendo: {e}")

else:
    st.write("Captura un registro de ejemplo:")

    col1, col2, col3 = st.columns(3)
    with col1:
        dateInput = st.date_input(label="Date", format="DD/MM/YYYY")
        time = st.time_input(label="Time", value=None)
        ride_distance = st.number_input("Ride Distance (km)", min_value=0.0, value=12.0, step=0.5)
        booking_value = st.number_input("Booking Value", min_value=0.0, value=25.0, step=1.0)
    with col2:
        avg_ctat = st.number_input("Avg CTAT (min)", min_value=0.0, value=20.0, step=1.0)
        avg_vtat = st.number_input("Avg VTAT (min)", min_value=0.0, value=8.0, step=1.0)
        vehicle_type = st.selectbox('Vehicle Type',
                                    ["Auto", "Go Sedan", "bike", "eBike", "Go Mini", "Premier Sedan", "Uber XL"])
        cancelledByCustomer = st.radio("Cancelled Rides by Customer", ["Si", "No"], horizontal=True)
    with col3:
        cancelledByRider = st.radio("Cancelled Rides by Rider", ["Si", "No"], horizontal=True)
        incompleteRides = st.radio("Incomplete Rides", ["Si", "No"], horizontal=True)
        payment_method = st.selectbox("Payment Method", ["Cash", "CreditCard", "DebitCard", "UberWallet", "UPI"],
                                      index=1)

    df_single = pd.DataFrame([{
        'Date': dateInput,
        'Time': time,
        "Ride Distance": ride_distance,
        "Booking Value": booking_value,
        "Avg CTAT": avg_ctat,
        "Avg VTAT": avg_vtat,
        "Payment Method": payment_method,
        'Vehicle Type': vehicle_type,
        'Cancelled Rides by Customer': cancelledByCustomer,
        'Reason for cancelling by Customer': '',
        'Cancelled Rides by Driver': cancelledByRider,
        'Driver Cancellation Reason': '',
        'Incomplete Rides': incompleteRides,
        'Incomplete Rides Reason': '',
        'Booking Status': ''
    }])

    if st.button("🔮 Predecir registro"):
        X = prepare_features(df_single)
        try:
            y_pred = modelSVM.predict(X)
            proba = None
            if hasattr(modelSVM, "predict_proba"):
                try:
                    proba = modelSVM.predict_proba(X)
                except Exception:
                    proba = None
            out = df_single.copy()
            out['Prediccion SVM'] = labelencoder.inverse_transform(y_pred)
            out = out.drop(
                ['Reason for cancelling by Customer', 'Driver Cancellation Reason', 'Incomplete Rides Reason'], axis=1)
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar resultados (CSV)", data=csv, file_name="predicciones_uber.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"Ocurrió un error prediciendo: {e}")
