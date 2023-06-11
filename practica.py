from scipy.special import factorial,comb
from scipy.stats import hypergeom,binom,expon,norm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics
#funcion de distribucion empirica 
import statsmodels.api as sm

def facto(n):
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
    datos_hombres = np.random.randint(0, 100, size=(n, 1))  # Generar n valores aleatorios entre 0 y 100 para hombres
    datos_mujeres = np.random.randint(0, 100, size=(n, 1))  # Generar n valores aleatorios entre 0 y 100 para mujeres

    # Crear dataframe
    df = pd.DataFrame({'Hombres': datos_hombres.flatten(), 'Mujeres': datos_mujeres.flatten()})

    # Escalar los datos a la suma de H y M
    df['Hombres'] = df['Hombres'] / df['Hombres'].sum() * H
    df['Mujeres'] = df['Mujeres'] / df['Mujeres'].sum() * M
    return df

def prob_mujer (tabla,i):
    cant_mujeres = tabla['Mujeres'][i]
    cant_total = 0
    for columna in tabla:
        cant_total += tabla[columna][i]
    return float(cant_mujeres/cant_total)


def prob_hipergeometrica(N,n,k):
    x= list(range(k+1)) #→ porque hasta k+1?
    cdf = hypergeom.cdf(x, N, n, k)#→ devuelve un arreglo con las acumuladas?
    tabla = pd.DataFrame({'x':x, 'f(x)':cdf})
    return tabla


def var_bernoulli ():
    #es suficiente?
    return random.random()

def dist_binomial(n,p,size):
    distribucion = binom(n,p)
    resultado = distribucion.rvs(size = size)  # Generar un número binomial aleatorio→ por qué?
    return resultado

def dist_expo(lambd,size):
# implementar una función que permita generar un número aleatorio con distribución
# Exp(λ)
    return expon.rvs(scale=1/lambd, size =size)

def dist_normal(m,sigma,size):
    return norm.rvs(loc  = m,scale= sigma, size = size)

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


def prob_acum_xy(tabla,n,m,k):
    result = 0
    for i in n:
        for x in m:
            if (i+x)<= k:
                result += tabla [x-1][i-1]
    return result

def prob_esperanza_xdadoy (tabla,n,m,k):
    results = []
    for i in n: #valores en x
        acum_x = 0
        for x in m: #valores en y
            #apendeo en la posicion x de los resultados, su proba puntual dado y m
            acum_x+= tabla [x-1][i-1]
        results.append(acum_x)

    return results

def esperanza_varianza_muestral(muestra):
    return statistics.mean(muestra),statistics.variance(muestra)

def histo_fr_relativas (muestra,ancho,title):
    plt.xlabel(r'$Y$',fontsize = 15)
    plt.ylabel(r'Frecuencia relativa',fontsize = 15)
    plt.hist(muestra, bins = np.arange(min(muestra),max(muestra)+ancho,ancho),weights=np.zeros(len(muestra))+1./len(muestra))
    plt.title (title)
    plt.show()
    
def func_dist_empirica(muestra):
    return sm.distributions.ECDF(muestra)

def muestras_independientes(n):
    return dist_normal(100,5,n)

def intervalo_confianza(muestra,varianza,confianza):
    confianza = confianza/100
    n = len(muestra)
    alpha = (1 - confianza) / 2
    media_muestra = esperanza_varianza_muestral(muestra)[0]
    z = norm.ppf(1 -alpha)  # Estadístico de la distribución normal estándar
    margen_error = z * np.sqrt(varianza / n)
    limite_inferior = media_muestra - margen_error
    limite_superior = media_muestra + margen_error
    return limite_inferior, limite_superior

def t_student (muestra,confianza):
    varianza_estimada = esperanza_varianza_muestral(muestra)[1]
    return intervalo_confianza(muestra,varianza_estimada,confianza)

def longitud_intervalo (a,b):
    return np.abs(b-a)

def main():
    np.random.seed(123)
    print("----GUIA 1----\ncalcular un factorial->",facto(5),'\n')
    print("----GUIA 2----\nmonito ->",monito('papaya'),'\n')
    tabla = tabla_hombres_mujeres(3,20,10)
    print("----GUIA 3----\n",tabla,'\n')
    print ("prob de ser mujer ->",prob_mujer(tabla,2),'\n')
    print("----GUIA 4 ----\nprob acumulada con dist hipergeométrica:\n",prob_hipergeometrica(10,3,2),'\n')
    print("----GUIA 5 ----\nnumero aleatorio con distribución binomial:\n",dist_binomial(10,0.5,1),'\n')
    print("----GUIA 5 ----\nnumero aleatorio con distribución exponencial:\n",dist_expo(0.5,1),'\n')
    print("----GUIA 5 ----\nnumero aleatorio con distribución normal:\n",dist_normal(1,2,1),'\n')
    variables = [0,1,2,3,4,5,6,7,8,9,10]
    probabilidades = [0.002,0.001,0.002,0.005,0.02,0.04,0.18,0.37,0.25,0.12,0.01]
    print("----GUIA 6 ----\n esperanza de una muestra:",esperanza(variables,probabilidades),'\n')
    print("----GUIA 6 ----\nvarianza en una muestra:",varianza(variables,probabilidades),'\n')

    tabla_prob_conj = [[0.05,0.05,0.1],
                       [0.05,0.1 ,0.35],
                       [0   ,0.2 ,0.1]]
    print("----GUIA 8 ----\n P(X+Y<2):",prob_acum_xy(tabla_prob_conj,[1,2,3],[1,2,3],2),'\n')
    results = prob_esperanza_xdadoy(tabla_prob_conj,[1,2,3],[1,2,3],2)
    for i in range(len(results)):
        print("----GUIA 8 ----\n E(X| Y=:",i+1,')→ ',results[i],'\n')
    datos = [7.3,8.6,10.4,16.1,12.2,15.1,14.5,9.3]
    esperanza_m,varianza_m = esperanza_varianza_muestral(datos)
    print("----GUIA 10 ----\n Esperanza muestral :",esperanza_m,'\n')
    print(" Varianza muestral :",varianza_m,'\n')
    muestra_expo_1 = dist_expo(0.5,10) 
    esp1,var1 = esperanza_varianza_muestral(muestra_expo_1)
    muestra_expo_2 = dist_expo(0.5,30)
    esp2,var2 = esperanza_varianza_muestral(muestra_expo_2)
    muestra_expo_3 = dist_expo(0.5,100)
    esp3,var3 = esperanza_varianza_muestral(muestra_expo_3)
    print("----GUIA 10 ----\n Muestra exponencial 10 elementos esperanza media:",esp1,'\n')
    print(" Muestra exponencial 10 elementos varianza:",var1,'\n')
    print("----GUIA 10 ----\n Muestra exponencial 30 elementos esperanza media:",esp2,'\n')
    print(" Muestra exponencial 30 elementos varianza:",var2,'\n')
    print("----GUIA 10 ----\n Muestra exponencial 100 elementos media:",esp3,'\n')
    print(" Muestra exponencial 100 elementos varianza:",var3,'\n')
    # for i in [0.4,0.2,0.1]:
    #     histo_fr_relativas(muestra_expo_1,i,'muestra con 10 elementos y ancho de banda'+str(i))
    #     histo_fr_relativas(muestra_expo_2,i,'muestra con 30 elementos y ancho de banda'+str(i)) 
    #     histo_fr_relativas(muestra_expo_3,i,'muestra con 100 elementos y ancho de banda'+str(i)) 
    print("----GUIA 10 ----\n Funcion de distribución empirica de una muestra exponencial:",func_dist_empirica(muestra_expo_1),'\n')
    muestra_normal = dist_normal(0,1,100)
    ecdf=func_dist_empirica(muestra_normal)
    muestra_uniforme = np.random.rand(100)
    muestra_nueva = ecdf(muestra_uniforme)
    print("----GUIA 10 ----\n Muestra de 100 datos ~N(0,1):",muestra_normal,'\n')
    print(" Muestra de 100 datos ~U(0,1) a los que se le aplicó la funcion de distribución empirica:\n",muestra_normal,'\n')
    muestra = muestras_independientes(100)
    for i in [95,98]:
        int_estimador = intervalo_confianza(muestra,5,i)
        int_student = t_student(muestra,i)
        print("----GUIA 11 ----\n Intervalo con ",i," de confianza usando el estimador →:(",int_estimador[0],',',int_estimador[1],')\n')
        print(" → Longitud = ",longitud_intervalo(int_estimador[0],int_estimador[1]),'\n')
        print(" Intervalo con ",i," de confianza usando t de student →:(",int_student[0],',',int_student[1],')\n')
        print(" → Longitud = ",longitud_intervalo(int_student[0],int_student[1]),'\n')



    
    

    




if __name__ == '__main__':
    main()

