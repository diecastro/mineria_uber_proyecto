#%% md
# # Punto 4: Modelamiento, Evaluación e Interpretación
# 
# # Estudiantes:
# * Diego Castro Díaz
# * Isabella Orozco Jordán
# * Manuela Idárraga Gómez
# 
# 
# Este notebook desarrolla el **ciclo de modelamiento predictivo** sobre el dataset de Uber (`data.csv`)
#%%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
selected_data['Payment Method'] = selected_data['Payment Method'].astype('category');
selected_data.info()

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
#revisionj de fuga de informacion / correlacion

X = selected_data.drop(columns=['Booking Status'])
y = selected_data['Booking Status']

X_enc = X.copy()
for col in X_enc.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col].astype(str))

# Calcular importancia de cada variable respecto al target
mi = mutual_info_classif(X_enc, y, discrete_features='auto')

mi_scores = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)
print(mi_scores)
#%%
cols_fuga = [
    'Driver Cancellation Reason',
    'Cancelled Rides by Driver',
    'Reason for cancelling by Customer',
    'Cancelled Rides by Customer',
    'Incomplete Rides Reason',
    'Incomplete Rides'
]
#No sirven para predecir, porque estan construidas con la misma informacion que el target.
X = selected_data.drop(columns=cols_fuga + ['Booking Status'])
Y = selected_data['Booking Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)  #En regresion no es muestreo estratificado
Y_train.plot(kind='hist')
#%%
# Copia del dataset
X_enc = X.copy()

# DecisionTreeClassifier no trabajka con objetos o cats
for col in X_enc.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col].astype(str))

X_train, X_test, Y_train, Y_test = train_test_split(X_enc, Y, test_size=0.3, random_state=42)

#%%
#Arbol de clasificación
#class_weight='balanced' balancea automatico
model_dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20, max_depth=5, class_weight='balanced')
model_dt.fit(X_train, Y_train)

#Evaluación
Y_pred = model_dt.predict(X_test)
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))

#%%
cm = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labelencoder.classes_,
            yticklabels=labelencoder.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
#%%
plt.figure(figsize=(20, 20))
plot_tree(model_dt, feature_names=X_train.columns.values, class_names=labelencoder.classes_, rounded=True, filled=True)
plt.show()
#%%
#Método Perezoso
model_knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
model_knn.fit(X_train, Y_train)

#Evaluación
Y_pred = model_knn.predict(X_test)
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
smote = SMOTE(random_state=42)
X_res, Y_res = smote.fit_resample(X_train, Y_train)
print("Tamaño original:", X_train.shape, " → Balanceado con SMOTE:", X_res.shape)
#Necesito usar smote para balancear la clase, similar a lo del arbol

#Red neuronal
model_rn = MLPClassifier(activation="relu", hidden_layer_sizes=(25), learning_rate='constant',
                         learning_rate_init=0.02, momentum=0.3, max_iter=500, verbose=False, random_state=42)
model_rn.fit(X_res, Y_res)

#Evaluación
Y_pred = model_rn.predict(X_test)
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
#Bagging: Knn
modelo_base = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

model_bag = BaggingClassifier(modelo_base, n_estimators=10, max_samples=0.6)  #n_estimators=100
model_bag.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_bag.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
#Random Forest
model_rf = RandomForestClassifier(n_estimators=100, max_samples=0.7, criterion='gini',
                                  max_depth=None, min_samples_leaf=2)
model_rf.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_rf.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
# Se imprimen la importancia de las características
print('Importancia de las características')
for i, j in sorted(zip(X_train.columns, model_rf.feature_importances_)):
    print(i, j)
#%%
#AdaBoost:Adaptive Boosting
#Aplicando smote, no resulto :/
# smote = SMOTE(random_state=42)
# X_res, Y_res = smote.fit_resample(X_train, Y_train)

modelo_base = DecisionTreeClassifier(max_depth=1, random_state=42)
model_boos = AdaBoostClassifier(
    modelo_base,
    n_estimators=200,
    random_state=42
)
model_boos.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_boos.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))

# TODO: este esta jodido revisarlo


#%%
# Gradient Boosting

#tasa de aprendizaje controla el tamaño de la actualización de cada modelo (contribución de cada nuevo árbol)
model_gbc = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42)
model_gbc.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_gbc.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
# XGboost
model_xgb = xgb.XGBClassifier(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8  #enable_categorical=True,
)

model_xgb.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_xgb.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
#CatBoostClassifier
model_cat = CatBoostClassifier(iterations=100, depth=10, verbose=False,
                               cat_features=[])  #Variables categóricas
model_cat.fit(X_train, Y_train)

#Evaluación
Y_pred = model_cat.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
#Votacion hard
clasificadores = [('dt', model_dt), ('knn', model_knn), ('net', model_rn)]

model_vot_hard = VotingClassifier(estimators=clasificadores, voting='hard')
model_vot_hard.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_vot_hard.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
#votacion soft

clasificadores = [('dt', model_dt), ('knn', model_knn), ('net', model_rn)]
model_vot_soft = VotingClassifier(estimators=clasificadores, voting='soft', weights=[0.3, 0.4, 0.3])
model_vot_soft.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_vot_soft.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
# stacking
clasificadores = [('dt', model_dt), ('knn', model_knn), ('net', model_rn)]

metodo_ensamblador = LogisticRegression()  #SVM, NN, KNN

model_stack = StackingClassifier(estimators=clasificadores, final_estimator=metodo_ensamblador)
model_stack.fit(X_train, Y_train)  #70%

#Evaluación
Y_pred = model_stack.predict(X_test)  #30%
print(metrics.classification_report(y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
filename = 'Modelo_Regresion_UBER.pkl'
variables = X.columns._values
classes = labelencoder.classes_
pickle.dump([model_gbc, variables, min_max_scaler, classes], open(filename, 'wb'))