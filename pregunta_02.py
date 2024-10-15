import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('aids_clinical_1-1.csv', sep=';')

#Definir variables
x = data[['preanti']]  #predictora
y = data['str2']  #predecir

#Dividir los datos en 85% para entrenamiento y 15% para prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

#Crear y entrenar el modelo
model = LinearRegression()
model.fit(x_train, y_train)

#Realizar predicciones
y_pred_train = model.predict(x_train)

#Evaluar modelo
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

#Resultados
print("Coeficiente de regresión (pendiente):", model.coef_)
print("Intersección:", model.intercept_)
print("Error Cuadrático Medio (MSE) - Train:", mse_train)
print("Puntaje de Varianza (R^2) - Train:", r2_train)