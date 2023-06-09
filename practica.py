from scipy.special import factorial,comb
from scipy.stats import hypergeom,binom,expon,norm
import pandas as pd
import numpy as np
import random


def facto(n):
    #calcular el factorial
    return factorial(n)

def consonantes(letras):
    cons =[]
    vocales = 'aeiouAEIOU'
    for letra in letras:
        if letra not in vocales:
            cons.append(letra)
    return cons

def monito (letras):
#devuelva la probabilidad de que el mono forme una palabra que no tenga vocales.
    cons =consonantes(letras)
    cant_const = len(cons)
    #Proba de ser consonante
    prob_letra = 1/cant_const
    prob_total = 0
    for i in range (cant_const):
        #cantidad de consonantes que sacamos
        prob_total+= comb(cant_const,i)/comb(len(letras),i)
    return prob_letra*prob_total

def tabla_hombres_mujeres(n,H,M):
# Crear una función cuyos inputs sean n, H y M y devuelva un dataframe con datos generados de
# manera aleatoria. La tabla tiene n filas, la fila i-ésima corresponde al nivel de estudio i-ésimo, y
# dos columnas una de hombres y otra de mujeres.
    # Generar datos aleatorios
    datos_hombres = np.random.randint(0, 100, size=(n, 1))  # Generar n valores aleatorios entre 0 y 100 para hombres
    datos_mujeres = np.random.randint(0, 100, size=(n, 1))  # Generar n valores aleatorios entre 0 y 100 para mujeres

    # Crear dataframe
    df = pd.DataFrame({'Hombres': datos_hombres.flatten(), 'Mujeres': datos_mujeres.flatten()})

    # Escalar los datos a la suma de H y M
    df['Hombres'] = df['Hombres'] / df['Hombres'].sum() * H
    df['Mujeres'] = df['Mujeres'] / df['Mujeres'].sum() * M
    return df

def prob_mujer (tabla,i):
# Crear una función que tenga de input una tabla como la del ejercicio a) y devuelva la probabilidad
# de ser mujer dado que tengo el estudio mas alto (el n-ésimo).
    cant_mujeres = tabla['Mujeres'][i]
    cant_total = 0
    for columna in tabla:
        cant_total += tabla[columna][i]
    return float(cant_mujeres/cant_total)


def prob_hipergeometrica(N,n,k):
# Hacer una función que calcule la función de distribución acumulada de una variable aleatoria Hiper-
# geométrica. Los inputs son los parámetros N, n, k y el output es un dataframe con 2 columnas, en una
# los rangos de x y en la otra F X (x) para los x en ese rango.
    x= list(range(k+1)) #→ porque hasta k+1?
    cdf = hypergeom.cdf(x, N, n, k)#→ devuelve un arreglo con las acumuladas?
    tabla = pd.DataFrame({'x':x, 'f(x)':cdf})
    return tabla


def var_bernoulli ():
    #es suficiente?
    return random.random()

def dist_binomial(n,p):
# genera un número binomial con los parámetros n, p.
    distribucion = binom(n,p)
    resultado = distribucion.rvs()  # Generar un número binomial aleatorio→ por qué?
    return resultado

def dist_expo(lambd):
# implementar una función que permita generar un número aleatorio con distribución
# Exp(λ)
    return expon.rvs(scale=1/lambd)

def dist_normal(m,sigma):
    return norm.rvs(loc  = m,scale= sigma)

def esperanza(var, prob):
    mean=0
    for i in range (len(prob)):
        mean+= var[i]*prob[i]
    return mean

def varianza(var,prob):
    mean = esperanza(var,prob)
    varianza = 0
    for i in range (len(var)):
        varianza+= ((var[i]-mean)**2)*prob[i]
    return varianza


def tabla_prob_conjunta(tabla):
   
def main():
    np.random.seed(123)
    print("----GUIA 1----\ncalcular un factorial->",facto(5),'\n')
    print("----GUIA 2----\nmonito ->",monito('papaya'),'\n')
    tabla = tabla_hombres_mujeres(3,20,10)
    print("----GUIA 3----\n",tabla,'\n')
    print ("prob de ser mujer ->",prob_mujer(tabla,2),'\n')
    print("----GUIA 4 ----\nprob acumulada con dist hipergeométrica:\n",prob_hipergeometrica(10,3,2),'\n')
    print("----GUIA 5 ----\nnumero aleatorio con distribución binomial:\n",dist_binomial(10,0.5),'\n')
    print("----GUIA 5 ----\nnumero aleatorio con distribución exponencial:\n",dist_expo(0.5),'\n')
    print("----GUIA 5 ----\nnumero aleatorio con distribución normal:\n",dist_normal(1,2),'\n')
    variables = [0,1,2,3,4,5,6,7,8,9,10]
    probabilidades = [0.002,0.001,0.002,0.005,0.02,0.04,0.18,0.37,0.25,0.12,0.01]
    print("----GUIA 6 ----\n esperanza de una muestra:",esperanza(variables,probabilidades),'\n')
    print("----GUIA 6 ----\nvarianza en una muestra:",varianza(variables,probabilidades),'\n')

    tabla_prob_conjunta = [0.05,0.05,0.01,
                           0.05,0.1,0.35,
                           0,0.2,0.1]
  




if __name__ == '__main__':
    main()