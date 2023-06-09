import math 
import statistics
import pandas as pd
from pylab import *
#Hacer una función que reciba de input una lista (la muestra) y devuelva la
# esperanza y la varianza muestral.

def media_muestral(muestra):
    result = 0
    for val in muestra:
        result+= val
    return result/len(muestra)
    # return statistics.mean(muestra)

def varianza_muestral (muestra):
    result = 0
    media = media_muestral(muestra)
    for val in muestra:
        result+= (val-media)**2
    return  result/(len(muestra)-1)
    #return statistics.variance(muestra)

def desviación_estandar(muestra):
    return varianza_muestral(muestra)**(1/2)
    #return statistics.stdev(muestra)

def ordenar_muestra(muestra):
    muestra.sort()
    return muestra

def generar_muestra(tamano):
    muestra = np.random.exponential(scale=1/0.5, size=tamano)
    return muestra

def plot_histogram(muestra, bin_width):
    plt.hist(muestra, bins =np.arange (min(muestra),max(muestra)+bin_width,bin_width),density = True)
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia Relativa')
    plt.title(f'Histograma con ancho de banda {bin_width}')
    plt.show()

def funcion_distribucion(muestra):
    muestra_ordenada = np.sort(muestra)
    n = len(muestra)
    valores_unicos, conteos = np.unique(muestra_ordenada, return_counts=True)
    prob_acumulada = np.cumsum(conteos) / n
    funcion_distribucion_empirica = np.concatenate(([0], prob_acumulada))
    return valores_unicos, funcion_distribucion_empirica

def main ():
    muestra= [3.4 ,2.5, 4.8, 2.9, 3.6,
                2.8, 3.3, 5.6, 3.7, 2.8,
                4.4, 4.0, 5.2, 3.0, 4.8]
    bin_widths = [0.4, 0.2, 0.1]

    print("tamaño de la muestra->"+str(len(muestra))+"\n")
    print ("media muestral -> "+ str(media_muestral(muestra))+"\n")
    print("varianza muestral ->" + str(varianza_muestral(muestra))+"\n")
    print("desviación estándar -> "+str(desviación_estandar(muestra))+"\n")
    # for bin in bin_widths:
    #     plot_histogram(muestra,bin)
    print("Función de distribución empírica:", funcion_distribucion(muestra))       
#     # muestra = ordenar_muestra(muestra)
#     # print("muestra ordenada -> ")
#     # print(muestra)
#     # print("\n")
# muestra_original = np.random.normal(0, 1, size=100) #-> generar una muestras N(0,1)


if __name__ == '__main__':
    main()