import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('breast_wisconsin_1.csv', sep=';')

#Definir variables
x = data[['symmetry3']] #predictora
y = data['fractal_dimension3'] #a predecir

#Dividir los datos en 70% para entrenamiento y 30% para prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x_train, y_train)

#Realizar predicciones en el conjunto de prueba
y_pred = model.predict(x_test)

#Evaluar el modelo con métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#resultados
print("Coeficiente de regresión (pendiente):", model.coef_)
print("Intersección:", model.intercept_)
print("Error Cuadrático Medio (MSE):", mse)
print("Puntaje de Varianza (R^2):", r2)