import pandas as pd
from scipy.stats import pointbiserialr

data = pd.read_csv('aids_clinical_1-1.csv', sep=';')

correlation, p_value = pointbiserialr(data['preanti'], data['homo'])

#Matriz de correlación
correlacion = data[['preanti', 'homo']].corr()
print("\nMatriz de correlación entre 'preanti' y 'homo':")
print(correlacion)

print(f"\nEl coeficiente de correlación punto-biserial entre preanti y homo es: {correlation:.3f}")

if correlation > 0:
    print("La correlación es positiva.")
    print("Significa que en promedio los valores de preanti tienden a ser más altos cuando homo es igual a 1 (presencia de la característica).")
elif correlation < 0:
    print("La correlación es negativa.")
    print("Significa que en promedio los valores de preanti tienden a ser más bajos cuando homo es igual a 1 (presencia de la característica).")
else:
    print("No hay correlación entre preanti y homo.")