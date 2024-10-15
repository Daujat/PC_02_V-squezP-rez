import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('state_x78-1.csv', sep=';')

data = data.dropna()

#variables predictora y objetivo
x = data[['ingresos']]
y = data['esp_vida']

#datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=42)

#objeto StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

#modelo de regresión lineal simple
model = LinearRegression()

#modelo con los datos de entrenamiento estandarizados
model.fit(x_train_scaled, y_train)

#datos de prueba utilizando el mismo escalador
x_test_scaled = scaler.transform(x_test)

y_pred = model.predict(x_test_scaled)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calcular la raíz del error cuadrático medio (RMSE)
rmse = np.sqrt(mse)

print("Coeficiente de regresión (pendiente):", model.coef_)
print("Intersección:", model.intercept_)
print("Error Cuadrático Medio (MSE) - Test:", mse)
print("Raíz Error Cuadrático Medio (RMSE) - Test:", rmse)