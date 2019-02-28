import numpy as np
import random
import math
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

###Data loading
def loadData(fileName):
    data=np.loadtxt(fileName)
    return data

k=20
eta=0.03
training=loadData("data/train.txt")
testing=loadData("data/test.txt")
M=int(max(training[:,0]))
N=int(max(training[:,1]))

i=training[:,0]
j=training[:,1]
yij=training[:,2]

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.10)

model = SVD(n_factors=20, n_epochs=100, biased=True, lr_all=0.03, 
                     reg_all=0.1,verbose=True)

model.fit(trainset)
    
U = model.pu
V = model.qi
ai = model.bu
bj = model.bi
    

V=V.T
V = V - np.transpose(np.array([np.mean(V, axis=1)]))
A,sig,B=np.linalg.svd(V)
A12=A[:,0:2]
#Uhat=np.matmul(np.transpose(A12),U)
Vhat=np.matmul(np.transpose(A12),V)
#any ten
#Vhat1=Vhat[:,0:10]
#plt.scatter(Vhat1[0,:],Vhat1[1,:])


#Ten most popular
popularlist=np.asarray([228,229,379,448,449,417,418,419,420])
title=['Star Trek III: The Search for Spock (1984)',
       'Star Trek IV: The Voyage Home (1986)',
       'Star Trek: Generations (1994)',
       'Star Trek: The Motion Picture (1979)',
       'Star Trek V: The Final Frontier (1989)',
       'Cinderella (1950)',
       'Mary Poppins (1964)',
       'Alice in Wonderland (1951)',
       'William Shakespeares Romeo and Juliet (1996)']
popular=[]
for i in popularlist:
    popular.append(Vhat[:,i])
popular=np.asarray(popular)
fig, ax = plt.subplots()
ax.scatter(popular[:,0],popular[:,1])
for txt in range(len(popularlist)):
    ax.annotate(title[txt],((popular[txt,0],popular[txt,1])))
ax.set_title('Different genre clusters example: Star Trek series and musical')