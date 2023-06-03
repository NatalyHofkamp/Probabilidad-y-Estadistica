import random
import math
# %%  En este ejercicio nos proponemos hacer un programa que resuelva el ejercicio 8 pero para una lista
# de letras cualquiera (sin repetir letras). Hacer una función que el input sea una lista de letras sin
# repetición y que devuelva la probabilidad de que el mono forme una palabra que no tenga vocales.

# Un mono tiene imanes de letras para formar palabras,
# (a) ¿Cuál es la probabilidad de que forme una palabra que no tenga vocales?
# (b) Si el mono usa todos los imanes que tiene, ¿cuál es la probabilidad de que arme una palabra que
# no tenga las dos vocales adyacentes?

def consonantes(letras):
    counter =0
    for letra in letras:
        if letra not in "aeiouAEIOU":
            counter +=1
    return counter

def monito (palabra):
    cons = consonantes(palabra)
    total = len(palabra)
    prob_letra = float (1/total)
    proba_total = 0
    for i in range(1,cons+1):
        proba_total += math.comb(cons, i)/math.comb(total,i)
    return prob_letra*proba_total
    
print(monito ('craiotuz'))

