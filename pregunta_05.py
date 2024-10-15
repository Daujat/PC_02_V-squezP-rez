import pandas as pd
import numpy as np

#DataFrame Original (Datos faltantes)
data = {
    'x1': [1, 2, 8, 4, 1, 6],
    'x2': [100, 2, 3, 1, 2, 7],
    'x3': [34, np.nan, np.nan, np.nan, 27, 44],
    'x4': [102, 121, 343, np.nan, 121, 125],
    'x5': [125, np.nan, 215, np.nan, 121, 125],
    'x6': [15, np.nan, 14, np.nan, 12, np.nan]
}

df = pd.DataFrame(data)

print("DataFrame original con datos faltantes:")
print(df)

#Imputar los datos faltantes con la mediana
df_imputado = df.fillna(df.median())

print("\nDataFrame con datos imputados utilizando la mediana:")
print(df_imputado)