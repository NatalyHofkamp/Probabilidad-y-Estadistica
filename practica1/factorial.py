#%%Implementarlo el factorial de las dos maneras y calcular el tiempo que tarde en cada forma 
# para n = 25, n = 100 y n = 200
import time
def recFact(num):
    if num==0:
        return 1;
    elif num<0:
        print("you entered a negative value.")
        return ValueError
    elif num==1:
        return 1
    else:
        return num*recFact(num-1)
    


def Fact(num):
    if num>0:
        for i in range(num-1,1,-1):
            num*=i;
    elif num==0:
        return 1;
    else:
        print("you entered a negative value.")
        return ValueError
    return num;


def P(n,r):
        return Fact(n)/Fact(n-r)

def main():
    start_time = time.time()
    #Fact(200)
    print(P(3,2))
    # elapsed_time = time.time() - start_time
    # print("Elapsed time: %0.10f seconds." % elapsed_time)
    recFact(4)
    # elapsed2_time = time.time() - start_time-elapsed_time
    # print("Elapsed time: %0.10f seconds." % elapsed2_time)
    return

if __name__ == '__main__':
    main()
# %%Implementar un código que calcule P (n, r) para todo n, r ≥ 0 entero.


