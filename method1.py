import numpy as np
import random
import math
import matplotlib.pyplot as plt

###Data loading
def loadData(fileName):
    data=np.loadtxt(fileName)
    return data
    
def grad_U(Ui, Yij, Vj, reg):
    uv=np.dot(Ui,Vj)
    gradU=reg*Ui-Vj*(Yij-uv)
    return gradU

def grad_V(Vj, Yij, Ui, reg):
    uv=np.dot(Ui,Vj)
    gradV=reg*Vj-Ui*(Yij-uv)
    return gradV

def get_err(U, V, Y, reg=0.0):
    err=0
    UV=np.asarray(np.matrix(np.transpose(U))*np.matrix(V))
    #forbU=np.linalg.norm(U)
    #forbV=np.linalg.norm(V)
    #regterm=(reg/2)*(forbU**2+forbV**2)
    for i in range(len(Y)):
        for d in range(len(Y[0])):
            if Y[i][d]>0.5: #Optimize run speed
                err=err+pow(Y[i][d]-UV[i][d],2)
    return err/2
    
def runSGD(Y,U,V,reg,eta,i,j):
    rand=random.randint(0,len(i)-1)
    Uind = int(i[rand]-1)
    Vind = int(j[rand]-1)
    U[:,Uind]=U[:,Uind]-eta*grad_U(U[:,Uind],Y[Uind][Vind],V[:,Vind],reg)
    V[:,Vind]=V[:,Vind]-eta*grad_V(V[:,Vind],Y[Uind][Vind],U[:,Uind],reg)  
    return

k=20
eta=0.03
training=loadData("data/train.txt")
testing=loadData("data/test.txt")
M=int(max(training[:,0]))
N=int(max(training[:,1]))

i=training[:,0]
j=training[:,1]
yij=training[:,2]
Y=[[0 for x in range(N)] for y in range(M)]
Y=np.asarray(Y,dtype = float)
for x in range(len(i)):
    Y[int(i[x])-1][int(j[x])-1]=yij[x]
itest=testing[:,0]
jtest=testing[:,1]
yijtest=testing[:,2]
Ytest=[[0 for x in range(N)] for y in range(M)]
Ytest=np.asarray(Ytest, dtype = float)  
for x in range(len(itest)):
    if itest[x]<= M and jtest[x] <= N:
        Ytest[int(itest[x])-1][int(jtest[x])-1] = yijtest[x]

#Method 1
E_in = []
E_out = []
iteration=0
U=[[random.uniform(-0.5,0.5) for x in range(M)] for y in range(k)]
V=[[random.uniform(-0.5,0.5) for x in range(N)] for y in range(k)]
U=np.asarray(U,dtype = float)
V=np.asarray(V,dtype = float)
los=[]
los.append(get_err(U,V,Y))
while (iteration/50000 <=300):
    iteration=iteration+1
    print("iteration %i" % iteration)
    runSGD(Y,U,V,0.1,eta,i,j)
    if iteration % 50000 == 0:
        newerr = get_err(U, V, Y)
        los.append(newerr)
        if (math.fabs(los[-1]-los[-2])/math.fabs(los[1]-los[0]) <= 0.0001):
            E_in.append(newerr/90000)
            E_out.append(get_err(U,V,Ytest)/10000)
            break

V = V - np.transpose(np.array([np.mean(V, axis=1)]))
A,sig,B=np.linalg.svd(V)
A12=A[:,0:2]
Uhat=np.matmul(np.transpose(A12),U)
Vhat=np.matmul(np.transpose(A12),V)
#any ten
Vhat1=Vhat[:,0:10]
plt.scatter(Vhat1[0,:],Vhat1[1,:])

#Ten most popular
popularlist=np.asarray([49,257,99,180,293,285,287,0,299,120])
popular=[]
for i in popularlist:
    popular.append(Vhat[:,i-1])
popular=np.asarray(popular)
plt.scatter(popular[:,0],popular[:,1])