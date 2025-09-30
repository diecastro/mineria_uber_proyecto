#%%
#importamos librerias básicas
import pandas as pd #manipulación dataframes
import numpy as np #matrices y vectores
import matplotlib.pyplot as plt #graficación
#%%
data=pd.read_excel("./data/Base de datos VIAJES UBER.xlsx")
data.head() #Listar las primeras filas de data
#%%
data.info()
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
#%%
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
#%%
data.info()
#%%
#Eliminar una columna
data = data.drop(['DateTime','fecha','hora','anio','mes','dia_semana','hora_cos','hora_sin'], axis=1)
data.head()
#%%
#Corrección de variables categóricas
data['Vehicle Type']=data['Vehicle Type'].astype('category')
data['Booking Status']=data['Booking Status'].astype('category')
data['Cancelled Rides by Customer']=data['Cancelled Rides by Customer'].astype('category')
data['Reason for cancelling by Customer']=data['Reason for cancelling by Customer'].astype('category')
data['Cancelled Rides by Driver']=data['Cancelled Rides by Driver'].astype('category')
data['Driver Cancellation Reason']=data['Driver Cancellation Reason'].astype('category')
data['Incomplete Rides']=data['Incomplete Rides'].astype('category')
data['Incomplete Rides Reason']=data['Incomplete Rides Reason'].astype('category')
data['Payment Method']=data['Payment Method'].astype('category')

data.info()
#%%
#Descripción de variables numéricas (para mirar si hay problemas de calidad de datos)
data.describe()
#%%
#Descripción variables categóricas
data['Vehicle Type'].value_counts().plot(kind='bar')
#%%
data['Booking Status'].value_counts().plot(kind='pie', autopct='%.0f%%')
#%%
# Cargar librería para Profiling
from ydata_profiling import ProfileReport
#%%
profile_data=ProfileReport(data, minimal=False) # minimal=True
profile_data
#%%
#Guardamos en html el perfilado de datos (para poder compartirlo como un informe en cualquier máquina)
profile_data.to_file(output_file="output.html")
#%%
#Se crean dummies a las variables predictoras categóricas (no a la variable obj)
data = pd.get_dummies(data, columns=['Vehicle Type','Reason for cancelling by Customer','Cancelled Rides by Driver','Driver Cancellation Reason','Incomplete Rides','Incomplete Rides Reason','Payment Method','Cancelled Rides by Customer'], drop_first=False, dtype=int)
data.head()
#%%
#Se codifican las categorias de la VARIABLE OBJETIVO

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data["Booking Status"]=labelencoder.fit_transform(data["Booking Status"]) #Objetivo

data.head()
#%%
#División 70-30
from sklearn.model_selection import train_test_split
X = data.drop("Booking Status", axis = 1) # Variables predictoras
Y = data['Booking Status'] #Variable objetivo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y) #Muestreo estratificado (mantiene la proporción de clases de la variable objetivo al dividr datos)
Y_train.value_counts().plot(kind='bar')
#%%
#Variable objetivo del 30%
Y_test.value_counts().plot(kind='bar')
#%%
X_train.head()
#%%
#Creación del modelo con el conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier

modelTree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, max_depth=None) #gini, entropy
modelTree.fit(X_train, Y_train) #70% train

#%%
from sklearn.tree import plot_tree
plt.figure(figsize=(6,8))
plot_tree(modelTree, feature_names=X_train.columns.values, class_names=labelencoder.classes_, rounded=True, filled=True)
plt.show()
#%%
#Evaluación 30%
from sklearn import metrics

Y_pred = modelTree.predict(X_test) #30% Test
print(Y_pred)

#%%
#Exactitud
exactitud=metrics.accuracy_score(y_true=Y_test, y_pred=Y_pred)
print(exactitud)
#%%
#Matriz de confusion
from sklearn import metrics

cm=metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred)
cm
#%%
#Plot de la matriz de confusion
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labelencoder.classes_)
disp.plot()
#%%
#Precision, Recall, f1, exactitud
print(metrics.classification_report( y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#%%
#Normalizacion las variables numéricas (las dummies no se normalizan)
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data[['Avg VTAT']]) #Ajuste de los parametros: max - min

#Se aplica la normalización a 70%  y 30%
X_train[['Avg VTAT']]= min_max_scaler.transform(X_train[['Avg VTAT']]) #70%
X_test[['Avg VTAT']]= min_max_scaler.transform(X_test[['Avg VTAT']]) #30%
#X_train[['Avg CTAT']]= min_max_scaler.transform(X_train[['Avg CTAT']]) #70%
#X_test[['Avg CTAT']]= min_max_scaler.transform(X_test[['Avg CTAT']]) #30%
#X_train[['Booking Value']]= min_max_scaler.transform(X_train[['Booking Value']]) #70%
#X_test[['Booking Value']]= min_max_scaler.transform(X_test[['Booking Value']]) #30%
#X_train[['Ride Distance']]= min_max_scaler.transform(X_train[['Ride Distance']]) #70%
#X_test[['Ride Distance']]= min_max_scaler.transform(X_test[['Ride Distance']]) #30%
#X_train.head()
#%%
#Normalizacion las variables numéricas (las dummies no se normalizan)
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data[['Avg CTAT']]) #Ajuste de los parametros: max - min

#Se aplica la normalización a 70%  y 30%
#X_train[['Avg VTAT']]= min_max_scaler.transform(X_train[['Avg VTAT']]) #70%
#X_test[['Avg VTAT']]= min_max_scaler.transform(X_test[['Avg VTAT']]) #30%
X_train[['Avg CTAT']]= min_max_scaler.transform(X_train[['Avg CTAT']]) #70%
X_test[['Avg CTAT']]= min_max_scaler.transform(X_test[['Avg CTAT']]) #30%
#X_train[['Booking Value']]= min_max_scaler.transform(X_train[['Booking Value']]) #70%
#X_test[['Booking Value']]= min_max_scaler.transform(X_test[['Booking Value']]) #30%
#X_train[['Ride Distance']]= min_max_scaler.transform(X_train[['Ride Distance']]) #70%
#X_test[['Ride Distance']]= min_max_scaler.transform(X_test[['Ride Distance']]) #30%
#X_train.head()
#%%
#Normalizacion las variables numéricas (las dummies no se normalizan)
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data[['Booking Value']]) #Ajuste de los parametros: max - min

#Se aplica la normalización a 70%  y 30%
#X_train[['Avg VTAT']]= min_max_scaler.transform(X_train[['Avg VTAT']]) #70%
#X_test[['Avg VTAT']]= min_max_scaler.transform(X_test[['Avg VTAT']]) #30%
#X_train[['Avg CTAT']]= min_max_scaler.transform(X_train[['Avg CTAT']]) #70%
#X_test[['Avg CTAT']]= min_max_scaler.transform(X_test[['Avg CTAT']]) #30%
X_train[['Booking Value']]= min_max_scaler.transform(X_train[['Booking Value']]) #70%
X_test[['Booking Value']]= min_max_scaler.transform(X_test[['Booking Value']]) #30%
#X_train[['Ride Distance']]= min_max_scaler.transform(X_train[['Ride Distance']]) #70%
#X_test[['Ride Distance']]= min_max_scaler.transform(X_test[['Ride Distance']]) #30%
#X_train.head()
#%%
#Normalizacion las variables numéricas (las dummies no se normalizan)
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data[['Ride Distance']]) #Ajuste de los parametros: max - min

#Se aplica la normalización a 70%  y 30%
#X_train[['Avg VTAT']]= min_max_scaler.transform(X_train[['Avg VTAT']]) #70%
#X_test[['Avg VTAT']]= min_max_scaler.transform(X_test[['Avg VTAT']]) #30%
#X_train[['Avg CTAT']]= min_max_scaler.transform(X_train[['Avg CTAT']]) #70%
#X_test[['Avg CTAT']]= min_max_scaler.transform(X_test[['Avg CTAT']]) #30%
#X_train[['Booking Value']]= min_max_scaler.transform(X_train[['Booking Value']]) #70%
#X_test[['Booking Value']]= min_max_scaler.transform(X_test[['Booking Value']]) #30%
X_train[['Ride Distance']]= min_max_scaler.transform(X_train[['Ride Distance']]) #70%
X_test[['Ride Distance']]= min_max_scaler.transform(X_test[['Ride Distance']]) #30%
#X_train.head()
#%%
#Aprendizaje KNN con 70%
from sklearn.neighbors  import KNeighborsClassifier #KNeighborsRegressor

modelKnn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')#euclidean, minkowski
modelKnn.fit(X_train, Y_train) #70%
#%%
#Evaluación de Knn con 30%
from sklearn import metrics

Y_pred = modelKnn.predict(X_test) #30%

#Matriz de confusion
cm=metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labelencoder.classes_)
disp.plot()

#Precision, Recall, f1, exactitud
print(metrics.classification_report( y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))

# Curva ROC - Removed as this is for binary classification, not multi-class
# metrics.RocCurveDisplay.from_estimator(modelKnn,X_test, Y_test)
#%%
#Red Neuronal

from sklearn.neural_network import MLPClassifier #MLPRegressor

#Solo se configura capas ocultas, no se configura capa de entrada y de salida
#activation -> función activación de la oculta
#hidden_layer_sizes=5,7 -> dos capas ocultas con 5 neuronas y 7 neuronas
#learning_rate-> tamaño del paso constante o decreciente
#learning_rate_init-> valor tasa de aprendizaje
#momentum->
#max_iter-> iteaciones
#random_state-> semilla para generacion numeros seudoaletorios
modelNN = MLPClassifier(activation="logistic",hidden_layer_sizes=(5), learning_rate='constant',
                     learning_rate_init=0.2, momentum= 0.3, max_iter=500, random_state=3)

modelNN.fit(X_train, Y_train) #70% normalizados
#%%
#Loss es la desviación entre Y_train y el Y_pred
#loss_curve_ es una lista con los valores de la función de pérdida (loss) en cada iteración/época durante el entrenamiento de la red neuronal.
loss_values = modelNN.loss_curve_
plt.plot(loss_values)
#%%
#Evaluación de Red Neuronal
from sklearn import metrics

Y_pred = modelNN.predict(X_test) #30%

#Matriz de confusion
cm=metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labelencoder.classes_)
disp.plot()

#Precision, Recall, f1, exactitud
print(metrics.classification_report( y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))

# Curva ROC - Removed as this is for binary classification, not multi-class
# metrics.RocCurveDisplay.from_estimator(modelNN,X_test, Y_test)
#%%
#SVM
from sklearn.svm import SVC # SVR

modelSVM = SVC(kernel='linear') #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
modelSVM.fit(X_train, Y_train) #70%

#%%
#Evaluación de SVM
from sklearn import metrics

Y_pred = modelSVM.predict(X_test) #30%

#Matriz de confusion
cm=metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labelencoder.classes_)
disp.plot()

#Precision, Recall, f1, exactitud
print(metrics.classification_report( y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))

# Curva ROC - Removed as this is for binary classification, not multi-class
# metrics.RocCurveDisplay.from_estimator(modelSVM,X_test, Y_test)
#%%
#pickle es una librería de Python para serializar objetos (guardarlos en un archivo en formato binario)

import pickle
filename = 'modelo.pkl'
variables= X.columns._values
pickle.dump([modelTree,modelKnn, modelNN, modelSVM, labelencoder,variables,min_max_scaler], open(filename, 'wb'))