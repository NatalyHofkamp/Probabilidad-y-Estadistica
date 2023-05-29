
#Hacer una función que calcule la función de distribución acumulada de una variable aleatoria Hiper-
# geométrica. Los inputs son los parámetros N, n, k y el output es un dataframe con 2 columnas, en una
# los rangos de x y en la otra F X (x) para los x en ese rango.
# Hint: primero hacer una función que calcule las probabilidades puntuales.
import pandas as pd 
def Fact(num):
    if num > 0:
        return num * Fact(num - 1)
    elif num == 0:
        return 1
    else:
        raise ValueError("Negative value not allowed")

def comb(n, r):
    if n < 0 or r < 0:
        raise ValueError("Negative value not allowed")
    return Fact(n) / (Fact(r) * Fact(n - r))

def print_tabla(tabla):
    for i in tabla:
        print(f'{i[0]} | {i[1]}')


def puntual_hipergeometrica(x, N, n, k):
    return ((comb(k, x) * comb(N - k, n - x)) / comb(N, n))

def distribucion_acumulada_hipergeometrica(N, n, k, rango):
    tabla = []
    for x in range(rango):
        prob = puntual_hipergeometrica(x, N, n, k)
        tabla.append([x, prob])
    
    # Calcular la acumulación de probabilidades
    acumulada = 0
    for i in range(len(tabla)):
        acumulada += tabla[i][1]
        tabla[i][1] = acumulada
    
    # Crear el dataframe con los resultados
    df = pd.DataFrame(tabla, columns=['x', 'F(x)'])
    return df

print_tabla(distribucion_acumulada_hipergeometrica(6, 3, 4,2))

