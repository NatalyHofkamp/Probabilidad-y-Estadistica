# Simular N variables aleatorias independientes con distribuci´on exponencial de par´ametro λ = 2 para
# N = 100, 200, 400, 600, 800. Calcular los intervalos de confianza del 90%, 95% y 99% para µ con
# varianza desconocida. Graficar los intervalos para cada confianza, en funci´on de N.

from scipy.stats import expon,norm,poisson,chi2
from statistics import mean, variance
import numpy as np 
import matplotlib.pyplot as plt 

# def dist_exp(lambd,size):
#     return expon.rvs(scale = 1/lambd , size = size)

# def media_varianza_muestral(muestra):
#     return mean(muestra),variance(muestra)

# def intervalo_confianza (muestra,conf):
#     media, S = media_varianza_muestral (muestra)
#     alpha = (1 - (conf/100))/2
#     Z= norm.ppf(1-alpha)
#     for i in range(len(muestra)):
#         sup= media + Z*(S/np.sqrt(len(muestra)))
#         inf= media - Z *(S/np.sqrt(len(muestra)))
#     return inf, sup

# def plot_int_conf(muestra, inf, sup):
#     mu, S = media_varianza_muestral(muestra)
#     plt.title("Intervalo de confianza para $\mu$\n", fontsize=20)
#     plt.ylabel("$(A, B)$", fontsize=15)
#     plt.xlabel("$N$", fontsize=15)
#     plt.plot((len(muestra),len(muestra)), (inf, sup), 'g-' if (mu >= inf and mu <= sup) else 'r-')


# def main():
#     for i in [100,200,400,600,800]:
#         for conf in [90,95,99]:
#             muestra = dist_exp (2,i)
#             inf,sup = intervalo_confianza(muestra,conf)
#             plot_int_conf(muestra,inf,sup)




# Simular 400 variables aleatorias independientes con distribuci´on exponencial de par´ametro λ = 2.
# Considerando la muestra calcular el intervalo de confianza del 95% para µ asumiendo varianza desconocida. 
# Volver a hacer esta simulaci´on y este intervalo 100 veces. Para cada simulaci´on fijarse si el
# # verdadero valor de µ est´a o no en el intervalo, mostrar esta informaci´on con alg´un gr´afico

# #------------------------------------------------------------------------------

# def dist_exp(lamb, size):
#     return expon.rvs(scale = 1/lamb, size = size)

# def media_var_muestral (muestra):
#     return mean(muestra),variance(muestra)

# def intervalo_confianza(muestra,conf):
#     alpha = (1-(conf/100))/2
#     Z = norm.ppf(1-alpha)
#     X,S = media_var_muestral(muestra)
#     inf = X - Z* (S/np.sqrt(len(muestra)))
#     sup = X + Z* (S/np.sqrt(len(muestra)))
#     return inf,sup

# def plot_somegraph(results,i):
#     plt.title('True Value of μ in Confidence Intervals')
#     plt.xlabel('Simulation')
#     plt.ylabel('True Value Inside Interval')
#     plt.plot(range(i+1), results, 'bo')
#     plt.show()

# def main():
#     results = []
#     no=0
#     for i in range (100):
#         muestra = dist_exp(2,400)
#         inf,sup = intervalo_confianza(muestra,95)
#         true_mean = 1/2
#         results.append(inf <= true_mean <= sup)
#         if((inf> true_mean) or (true_mean > sup) ):
#             no+=1
#     print ("Los intervalos tenian un fallo de "+str((no/len(muestra))*100)+"%")
#     plot_somegraph(results,i)
    
#-------------------------------------------------------------------------------------------------------------------------------
# Simular N (con N = 100, 200, 300, 400) v.a.i.i.d. con distribución Poisson de parámetro λ = 1/2. Para
# cada valor de N hacer los histogramas de frecuencias relativas sabiendo que la distribución es discreta
# y conociendo su rango. ¿Importa el ancho de banda?

# def  dist_poisson(lamb,size):
#     return poisson.rvs(mu = lamb, size = size)

# def histogramas (muestra, ancho):
#     plt.xlabel(r'$Y$',fontsize = 15)
#     plt.ylabel(r'Frecuencia relativa',fontsize = 15)
#     plt.title("Histograma de frecuencias relativas para ancho = "+str(ancho)+" con "+str(len(muestra))+" elementos")
#     plt.hist(muestra, bins = np.arange(min(muestra),max(muestra)+ancho, ancho), weights = np.zeros(len(muestra))+1/len(muestra))
#     plt.show()
# def main():
#     for n in [100,200,300,400]:
#         muestra = dist_poisson(1/2,n)
#         histogramas(muestra,1)
#         histogramas (muestra,0.5)

#------------------------------------------------------------------------------------------------------------------------
# Simular N (con N = 100, 200, 300, 400) v.a.i.i.d. con distribuci´on Uniforme en el intervalo [2, 5]. Para
# cada valor de N hacer los histogramas de probabilidad con anchos de banda 2, 1 y 0.5

# def dist_uni (min,max,size):
#     return np.random.uniform(low=min, high=max, size=size)

# def histogramas (muestra, ancho):
#     plt.hist(muestra, bins = np.arange(min(muestra),max(muestra)+ancho,ancho), weights = np.zeros(len(muestra))+1/len(muestra))
#     plt.title ("Frecuencias relativas para "+str(len(muestra))+" elementos con ancho de banda "+str(ancho))
#     plt.xlabel(r'$Y$', fontsize = 15)
#     plt.ylabel(r'$Frecuencias relativas$', fontsize = 15)
#     plt.show()

# def main():
#     for i in [100,200,300,400]:
#         for ancho in [2,1,0.5]:
#             muestra  = dist_uni(2,5,i)
#             histogramas(muestra,ancho)

#-------------------------------------------------------------------------------------------------------------------------------
# Simular N = 400 v.a.i.i.d. con distribuci´on normal µ = 2, σ = 1/2 y N v.a.i.i.d. con distribuci´on
# normal µ = −1 y σ = 1/4. Considerar los datos que vienen de sumar lugar a lugar cada una de estas
# simulaciones. De estos nuevos datos, calcular la media y la varianza muestral. Interpretar. Graficar
# el histograma de probabilidad con ancho de banda 0.5, 1 y 4. ¿Qu´e observa?

# def dist_norm(n,p,size):
#     return norm.rvs(loc=n, scale= p, size = size)

# def histogramas (muestra,ancho):
#     plt.hist(muestra,np.arange(min(muestra),max(muestra)+ancho, ancho), weights = np.zeros(len(muestra))+1/len(muestra), density= True)
#     plt.title("Probabilidad de una  muestra Normal de "+str(len(muestra))+ "elementos. Ancho de banda = "+str(ancho), fontsize = 20)
#     plt.xlabel(r'$Y$', fontsize = 15)
#     plt.ylabel(r'Frecuencias relativas', fontsize = 15)
#     plt.show()

# def media_var_muestral (muestra):
#     return mean(muestra),variance(muestra)

# def main():
#     muestra1= dist_norm(2,1/2,400)
#     muestra2 = dist_norm(-1,1/4,400)   
#     muestra = muestra1-muestra2 
#     media, varianza = media_var_muestral(muestra)
#     print("Media muestral →",media,'\n')
#     print ("Varianza muestral →",varianza,'\n')
#     for i in [0.5,1,4]:
#         histogramas(muestra,i)


#-------------------------------------------------------------------------------------------------------------------------------

# Simular 20 variables aleatorias independientes llamadas Xi, i = 1, 2, . . . , 20 cada una con distribucion normal de
# parametros µi = i y σ=1/i, i = 1, 2, . . . , 20. Sean U =la sumatoria de 1 a 20 de Xi y Z = U−E(U)/raiz(var(u)).
# Volver a realizar este procedimiento de manera independiente para obtener 100 muestras de la variable aleatoria Z. 
# Utilizando la muestra de Z graficar el histograma de probabilidad. Hint: Recordar la formula para
# la esperanza y la varianza de suma de normales, la cual esta en la clase de distribuciones especiales.

# def dist_norm (n,p):
#     return norm.rvs(loc = n, scale = p)

# def histo_proba (muestra,ancho):
#     plt.hist(muestra, np.arange(min(muestra),max(muestra)+ancho, ancho), weights = np.zeros(len(muestra))+1/len(muestra), density = True)
#     plt.title("Probabilidad de una normal con "+str(len(muestra))+ " elementos con ancho "+str(ancho))
#     plt.ylabel(r'$Y$')
#     plt.xlabel("Probabilidad")
#     plt.show()

# def generateZ (U, esp ,var):
#     return (U-esp)/np.sqrt(var)
    
# def main():
#     variables_normales = []
#     muestra_z = []
#     esp_u = 0
#     var_u = 0 
#     for i in range (1,21):
#         esp_u+= i
#         var_u += 1/i
#     for _ in range (100): 
#         for i in range (1,21):
#             variables_normales.append(dist_norm(i,1/i))
#         U = np.sum(variables_normales)
#         muestra_z.append(generateZ(U, esp_u, var_u))
#     return histo_proba(muestra_z,0.1)

#-----------------------------------------------------------------------------------------------------------
# Simular N = 200 v.a.i.i.d. una con distribución normal de parámetros μ = 2 y σ = 3. Usar estos datos
# para calcular un intervalo de confianza de nivel 90%, 95% y 99% para la varianza. ¿Cuál intervalo es
# # mas grande y cual mas chico? ¿A qué se debe esto?

# def dist_norm(n,p,size):
#     return norm.rvs(loc = n, scale =p, size = size)

# def med_var_media (muestra):
#     return mean(muestra),variance(muestra)

# def intervalo_confianza(muestra,confianza):
#     n = len(muestra)
#     alpha = (1-confianza/100)/2
#     media,var =  med_var_media(muestra)
#     chi2_upper = chi2.ppf(alpha, df=n-1)
#     chi2_lower = chi2.ppf(1 - alpha, df=n-1)
#     sup = (n-1) * var / chi2_upper
#     inf = (n-1) * var / chi2_lower
#     return inf,sup

# def plot_longitudes(muestra,inf,sup):
#     longitud = sup-inf
#     plt.plot(len(muestra),(longitud))


# def plot_intervalos(intervalos, conf):
#     plt.figure()
#     plt.title("Intervalo de confianza para $\mu$", fontsize=20)
#     plt.ylabel("$(A, B)$", fontsize=15)
#     plt.xlabel("$N$", fontsize=15)

#     for intervalo in intervalos:
#         plt.plot([intervalo[0], intervalo[1]], 'g-')

#     plt.show()


def read_data (filename):
    plt.figure() 
    ancho1 = []
    largo1 =[]
    ancho2 = []
    largo2 =[]
    with open(filename, 'r') as archivo_csv:
        lineas = archivo_csv.readlines()
        for linea in lineas:
            linea= linea.split().strip()
            print(linea)
            if linea[0]== "C1":
                largo1.append(int(linea[1]))
                ancho1.append(int(linea[-1]))
            if linea[0]=="C2":
                largo2.append(int(linea[1]))
                ancho2.append(int(linea[-1]))
    return largo1,ancho1,largo2,ancho2


def plot_histogramas(muestra):
    plt.hist(muestra,np.arange(min(muestra),max(muestra)+0.1, 0.1), weights = np.zeros(len(muestra))+1/len(muestra), density= True)
    plt.title("Probabilidad de una  muestra Normal de "+str(len(muestra))+ "elementos. Ancho de banda = "+str(0.1), fontsize = 20)
    plt.xlabel(r'$Y$', fontsize = 15)
    plt.ylabel(r'Frecuencias relativas', fontsize = 15)
    plt.show()


def main():
    largo1,ancho1,largo2,ancho2 = read_data('datos.csv')
    print (largo1)
    plot_histogramas(largo1)
    plot_histogramas(largo2)
    plot_histogramas(ancho1)
    plot_histogramas(ancho2)



if __name__ =='__main__':
    main()

    