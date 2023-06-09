import matplotlib.pyplot as plt
import numpy as np

filenames = ['afghanistan.csv', 'estados_unidos.csv', 'china.csv', 'india.csv', 'egipto.csv', 'arabia_saudita.csv']

plt.figure() 
for filename in filenames:
    pbi = []
    with open(filename, 'r') as archivo_csv:
        lineas = archivo_csv.readlines()
        for linea in lineas:
            pbi.append(linea[0])

    tiempo = np.arange(1950, 2017, 1)  
    filename = filename.split('.')
    plt.plot(tiempo, pbi, label=filename[0])

plt.xlabel('Año')
plt.ylabel('Valor')
plt.title('Evolución del PBI')
plt.legend()
plt.show()