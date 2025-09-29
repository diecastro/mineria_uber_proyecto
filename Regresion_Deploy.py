import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

filename = './Modelo_Regresion_UBER.pkl'
model_gbc, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

data = pd.read_csv("./data_prediccion.csv")
data.head()