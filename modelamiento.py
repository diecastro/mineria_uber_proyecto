#%% md
# # Punto 4: Modelamiento, Evaluación e Interpretación
# 
# 
# Este notebook desarrolla el **ciclo de modelamiento predictivo** sobre el dataset de Uber (`data.csv`)
#%%
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

#%%
data = pd.read_csv('./data/data.csv')
data.head()
#%%
target = 'Booking Status'
y = data[target].astype(str)
X = data.drop(columns=[target])

num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
cat_cols = [c for c in X.columns if c not in num_cols]
print("Variables numéricas:", num_cols)
print("Variables categóricas:", cat_cols)
#%%
def cambiodetipo(array, tipo):
    for each in array:
        data[each] = data[each].astype(tipo);
#%%
#Correccion de tipo de datos
variables_categoricas_a_corregir = ['Vehicle Type', 'Booking Status', 'Reason for cancelling by Customer',
                                    'Driver Cancellation Reason', 'Incomplete Rides Reason', 'Payment Method']
cambiodetipo(variables_categoricas_a_corregir, 'category')
cambiodetipo(['Pickup Location', 'Drop Location', 'Booking ID', 'Customer ID'], 'string')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
data.info()
#%%
min_max_scaler = MinMaxScaler()
variables_numericas = ['Cancelled Rides by Customer', 'Cancelled Rides by Driver', 'Incomplete Rides']
min_max_scaler.fit(data[variables_numericas])  #Ajuste de los parametros: max - min
data[variables_numericas] = min_max_scaler.transform(data[variables_numericas])
data.head()
#%%
#Eliminacion de variables
selected_data = data.copy()
selected_data = selected_data.drop(
    columns=['Booking ID', 'Customer ID', 'Vehicle Type', 'Pickup Location', 'Drop Location', 'Date', 'Time',
             'Customer Rating', 'Driver Ratings'])
selected_data.head()
#%%
#Reduccion de dimensionalidad
# Cash, NA, Debit
mapa_pago = {
    'DebitCard': 'Debit',
    'CreditCard': 'Debit',
    'UberWallet': 'Debit',
    'Cash': 'Cash',
    'UPI': 'Debit',
}
selected_data['Payment Method'] = selected_data['Payment Method'].map(mapa_pago)
selected_data.head()
#%%
# Transformacion de variables
distance_bins = [0, 15, 30, float('inf')]
distance_labels = ['Corta', 'Intermedia', 'Extensa']

selected_data['Ride Distance'] = pd.cut(selected_data['Ride Distance'], bins=distance_bins, labels=distance_labels,
                                        include_lowest=True)
selected_data['Ride Distance'].value_counts().plot(kind='bar')
#%%
# Transformacion de variables
ctat_bins = [0, 10, 45, float('inf')]
ctat_labels = ['Alto', 'Medio', 'Bajo']

selected_data['Avg CTAT'] = pd.cut(selected_data['Avg CTAT'], bins=ctat_bins, labels=ctat_labels, include_lowest=True)
selected_data['Avg CTAT'].value_counts().plot(kind='bar')
#%%
vtat_bins = [2, 7, 12, 20]
vtat_labels = ['Bajo', 'Medio', 'Alto']

selected_data['Avg VTAT'] = pd.cut(
    selected_data['Avg VTAT'],
    bins=vtat_bins,
    labels=vtat_labels,
    include_lowest=True
)
selected_data['Avg VTAT'].value_counts().plot(kind='bar')

#%%
#LabelEncoder para la variable objetivo
labelencoder = LabelEncoder()
selected_data["Booking Status"] = labelencoder.fit_transform(data["Booking Status"])
data.head()
#%%
X = selected_data.drop("Booking Status", axis=1)
Y = selected_data['Booking Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)  #En regresion no es muestreo estratificado
Y_train.plot(kind='hist')
#%%
#Arbol de clasificación
model_dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20, max_depth=5)
model_dt.fit(X_train, Y_train)

#Evaluación
Y_pred = model_dt.predict(X_test)
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))