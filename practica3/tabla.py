import random
#%%La idea de este ejercicio es generalizar el ejercicio 10 para una cantidad de niveles de estudios arbitraria
# n y cantidades arbitrarias de hombres H y mujeres M en la muestra. Por simplicidad vamos a permitir
# que los números de la tabla sean números no enteros (con coma).
# (a) Crear una función cuyos inputs sean n, H y M y devuelva un dataframe con datos generados de
# manera aleatoria. La tabla tiene n filas, la fila i-ésima corresponde al nivel de estudio i-ésimo, y
# dos columnas una de hombres y otra de mujeres.
# (b) Crear una función que tenga de input una tabla como la del ejercicio a) y devuelva la probabilidad
# de ser mujer dado que tengo el estudio mas alto (el n-ésimo).

def dataframe(n,H,M):
    tabla= []
    for i in range(n):
        generated_H = random.randrange(0,H,1)
        generated_M = random.randrange (0,M,1)
        tabla.append([generated_H,generated_M])
    return tabla;

def prob_be_woman(tabla):
    return (tabla[-1][1]/(tabla[-1][1]+tabla[-1][0]))

def print_table(table):
    for i in table:
        print(f'|{i[0]} | {i[1]}|')

tabla = dataframe(10,20,15)
print_table(tabla)
print(prob_be_woman(tabla))
