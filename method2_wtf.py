import numpy as np
import random
import math
import matplotlib.pyplot as plt

###Data loading
def loadData(fileName):
    data=np.loadtxt(fileName)
    return data
    
def grad_U(Ui, Yij, Vj, reg,ai,bj,mu=0):
    uv=np.dot(Ui,Vj)
    gradU=reg*Ui-Vj*(Yij-uv-ai-bj-mu)
    return gradU

def grad_V(Vj, Yij, Ui, reg,ai,bj,mu=0):
    uv=np.dot(Ui,Vj)
    gradV=reg*Vj-Ui*(Yij-uv-ai-bj-mu)
    return gradV

def grad_A(Ui, Yij, Vj, reg,ai,bj,mu=0):
    uv=np.dot(Ui,Vj)
    gradA=reg*ai-(Yij-uv-ai-bj-mu)
    return gradA

def grad_B(Ui, Yij, Vj, reg,ai,bj,mu=0):
    uv=np.dot(Ui,Vj)
    gradB=reg*bj-(Yij-uv-ai-bj-mu)
    return gradB

def grad_mu(Ui, Yij, Vj, reg,ai,bj,mu=0):
    uv=np.dot(Ui,Vj)
    gradmu=-(Yij-uv-ai-bj-mu)
    return gradmu

def get_err(U, V, Y, reg,a,b,mu,sample):
    err=0
    UV=np.asarray(np.matrix(np.transpose(U))*np.matrix(V))
    forbU=np.linalg.norm(U)
    forbV=np.linalg.norm(V)
    forbA=np.linalg.norm(a)
    forbB=np.linalg.norm(b)
    regterm=(reg/2)*(forbU**2+forbV**2+forbA**2+forbB**2)
    for i in range(len(Y)):
        for d in range(len(Y[0])):
            if Y[i][d]>0.5: #Optimize run speed
                err=err+pow(Y[i][d]-UV[i][d]-a[i]-b[d]-mu,2)
    err=err+regterm
    return np.sqrt(err/sample)
    
def runSGD(Y,U,V,reg,eta,i,j,a,b,mu=0):
    rand=random.randint(0,len(i)-1)
    Uind = int(i[rand]-1)
    Vind = int(j[rand]-1)
    U[:,Uind]=U[:,Uind]-eta*grad_U(U[:,Uind],Y[Uind][Vind],V[:,Vind],reg,a[Uind],b[Vind],mu)
    V[:,Vind]=V[:,Vind]-eta*grad_V(V[:,Vind],Y[Uind][Vind],U[:,Uind],reg,a[Uind],b[Vind],mu)  
    a[Uind]=a[Uind]-eta*grad_A(U[:,Uind],Y[Uind][Vind],V[:,Vind],reg,a[Uind],b[Vind],mu)
    b[Vind]=b[Vind]-eta*grad_B(U[:,Uind],Y[Uind][Vind],V[:,Vind],reg,a[Uind],b[Vind],mu)
    mu=mu-eta*grad_mu(U[:,Uind],Y[Uind][Vind],V[:,Vind],reg,a[Uind],b[Vind],mu)
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

#Method 2
E_in = []
E_out = []
iteration=0
U=[[random.uniform(-0.5,0.5) for x in range(M)] for y in range(k)]
V=[[random.uniform(-0.5,0.5) for x in range(N)] for y in range(k)]
U=np.asarray(U,dtype = float)
V=np.asarray(V,dtype = float)
a = np.random.uniform(-0.5,0.5,M)
b = np.random.uniform(-0.5,0.5,N)
mu=np.mean(yij)
los=[]
los.append(get_err(U,V,Y,0.1,a,b,mu,90000))
while (iteration/10000 <=300):
    iteration=iteration+1
    print("iteration %i" % iteration)
    runSGD(Y,U,V,0.1,eta,i,j,a,b,mu)
    if iteration % 10000 == 0:
        newerr = get_err(U, V, Y,0.1,a,b,mu,90000)
        los.append(newerr)
        if (math.fabs(los[-1]-los[-2])/math.fabs(los[1]-los[0]) <= 0.0001):
            E_in.append(newerr)
            E_out.append(get_err(U,V,Ytest,0.1,a,b,mu,10000))
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
title=['Star Wars (1977)','Contact (1997)','Lightning Jack (1994)','Low Down Dirty Shame, A (1994)','Ayn Rand: A Sense of Life (1997)','English Patient, The (1996)','Scream (1996)','Toy Story (1995)','Air Force One (1997)','Independence Day (ID4) (1996)']
popular=[]
for i in popularlist:
    popular.append(Vhat[:,i-1])
popular=np.asarray(popular)
fig, ax = plt.subplots()
ax.scatter(popular[:,0],popular[:,1])
for txt in range(len(popularlist)):
    ax.annotate(title[txt],((popular[txt,0],popular[txt,1])))