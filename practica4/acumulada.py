
#Hacer una función que calcule la función de distribución acumulada de una variable aleatoria Hiper-
# geométrica. Los inputs son los parámetros N, n, k y el output es un dataframe con 2 columnas, en una
# los rangos de x y en la otra F X (x) para los x en ese rango.
# Hint: primero hacer una función que calcule las probabilidades puntuales.
import pandas as pd 
import math

def print_tabla(tabla):
    for i in tabla:
        print(f'{i[0]} | {i[1]}')


def puntual_hipergeometrica(x, N, n, k):
    return (math.comb(k, x) * math.comb(N - k, n - x)) / math.comb(N, n)

def distribucion_acumulada_hipergeometrica(N, n, k, rango):
    tabla = []
    for x in range(rango):
        prob = puntual_hipergeometrica(x, N, n, k)
        tabla.append([x, prob])
    acumulada = 0
    for i in range(len(tabla)):
        acumulada += tabla[i][1]
        tabla[i][1] = acumulada
    df = pd.DataFrame(tabla, columns=['x', 'F(x)'])
    return df

print_tabla(distribucion_acumulada_hipergeometrica(6, 3, 4,2))

