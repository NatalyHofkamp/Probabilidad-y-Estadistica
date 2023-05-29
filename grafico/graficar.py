import matplotlib.pyplot as plt
import numpy as np

ruta_archivo = 'datos_pbi.csv'  # Reemplaza con la ruta y nombre de tu archivo CSV

consumo = []
pbi = []
inversion = []

with open(ruta_archivo, 'r') as archivo_csv:
    lineas = archivo_csv.readlines()
    i = 0
    for linea in lineas:
        linea = list(linea.strip().split(',,'))
        for año in linea:
            año = list(año.strip().split(','))
            if i == 0:
                pbi.append(float(año[0]))  # primer trimestre
                pbi.append(float(año[2]))  # tercer trimestre
            elif i == 1:
                consumo.append(float(año[0]))  # primer trimestre
                consumo.append(float(año[2]))  # tercer trimestre
            else:
                inversion.append(float(año[0]))  # primer trimestre
                inversion.append(float(año[2]))  # tercer trimestre
        i += 1

tiempo = np.arange(2013, 2023, 0.5)  # Arreglo de tiempo con trimestres

plt.plot(tiempo, consumo, label='Consumo', linestyle=':')
plt.plot(tiempo, inversion, label='Inversión',linestyle ='--')
plt.plot(tiempo, pbi, label='PBI')


plt.xlabel('Año')
plt.ylabel('Valor')
plt.title('Evolución del Consumo, Inversión y PBI')
plt.legend()
plt.show()