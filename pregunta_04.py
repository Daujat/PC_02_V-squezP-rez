import pandas as pd

data = pd.read_csv('ingreso-1.csv')

correlation = data['ingreso'].corr(data['horas'])

correlacion = data[['ingreso', 'horas']].corr()
print("\nMatriz de correlación entre 'ingreso' y 'horas':")
print(correlacion)

print(f"\nEl coeficiente de correlación de Pearson entre ingreso y horas es: {correlation:.3f}")

if correlation > 0:
    print("La correlación es positiva.")
    print("Significa que a medida que aumentan las horas trabajadas los ingresos tienden a aumentar.")
elif correlation < 0:
    print("La correlación es negativa.")
    print("Significa que a medida que aumentan las horas trabajadas los ingresos tienden a disminuir.")
else:
    print("No hay correlación entre ingreso y horas.")