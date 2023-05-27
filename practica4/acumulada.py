
#Hacer una función que calcule la función de distribución acumulada de una variable aleatoria Hiper-
# geométrica. Los inputs son los parámetros N, n, k y el output es un dataframe con 2 columnas, en una
# los rangos de x y en la otra F X (x) para los x en ese rango.
# Hint: primero hacer una función que calcule las probabilidades puntuales.
def Fact(num):
    if num > 0:
        return num * Fact(num - 1)
    elif num == 0:
        return 1
    else:
        raise ValueError("Negative value not allowed")

def comb(n, r):
    return Fact(n) / (Fact(r) * Fact(n - r))


def print_tabla(tabla):
    for i in tabla:
        print(f'{i[0]} | {i[1]}')


def hipergeometrica(x, N, n, k):
    return ((comb(k, x) * comb(N - k, n - x)) / comb(N, n))


def acumulada(N, n, k):
    tabla = []
    for i in range(n):
        tabla.append([i, hipergeometrica(i, N, n, k)])
    return tabla


print_tabla(acumulada(6, 3, 4))

