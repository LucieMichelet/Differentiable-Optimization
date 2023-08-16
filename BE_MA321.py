# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 18:20:41 2022

@author: Lucie
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation
from tqdm import tqdm


#%% Bibliothèque 

#Méthode du gradient conjugué
def GPC(A,b,x0,epsilon) : 
    x=copy.copy(x0)
    d=b-A@x
    compteur=0
    y=x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) >epsilon and compteur <1000 : 
         y=copy.copy(x)
         t=-(d.T @ (A@x-b))/(d.T @A @d)
         x=x+t*d
         beta=(d.T @ A @ (A@x-b)) / (d.T @ A @ d)
         d=-(A@x-b)+beta*d
         compteur+=1
    return x  


#Définition du projeté
def proj(x):
    p = np.maximum(np.minimum(x,0.5*np.ones(np.shape(x))),-0.5*np.ones(np.shape(x)))
    return p
 

#Méthode du gradient projeté
def Gproj(A,b,rho,x0,epsilon) : 
    x=copy.copy(x0)
    r=b-A@x
    compteur=0
    y=x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) >epsilon and compteur <1000:
        y=copy.copy(x) 
        x=proj(x+rho*r) 
        r=b-A@x
        compteur+=1
    return x


#%% -----------------------------------------Initialisation des données------------------------------------

M = 200
N = 300
t0 = 0
tf = 10
c = 1
L = 1
teta = 1
dt=tf/(N+1)
dx=1/(M+1)
beta = (c*(dt/dx))**2
x=np.linspace(dx,1-dx,M)

def u0(x):
   return 0.5*np.sin(np.pi*x).reshape(len(x),1) 

def v0(x):
     return -4*np.sin(np.pi*x).reshape(len(x),1) 
 

#Definition des matrices P0, P1 et P2
P2 = np.eye(M)+np.eye(M)*beta*teta-(beta*teta)/2*np.diag(np.ones(M-1),1)-(beta*teta)/2*np.diag(np.ones(M-1),-1)
P0=-P2 
P1 = -(-2*((1+beta*(1-teta))*np.eye(M))-beta*(1-teta)*np.diag(np.ones(M-1),1)-beta*(1-teta)*np.diag(np.ones(M-1),-1))

#Définition de U0 et U1
A=2*np.eye(M)-np.diag(np.ones(M-1),1)-np.diag(np.ones(M-1),-1)
U0 = u0(x)
U1 = U0+dt*v0(x)+0.5*(dt**2)*(np.zeros((len(x),1))-((c/dx)**2)*A@U0)

U=np.zeros((M,N+2))
U[:,0]=U0.reshape(M)
U[:,1]=U1.reshape(M)



#%% Déduire un+2 avec le Gradient Conjugué

for n in tqdm(range(N)) : 
    b = P1@U[:,n+1] +P0@U[:,n]                                                  #on calcul b
    U2 = GPC(P2,b,U[:,n],10**(-10)).reshape(200,1)                              #on calcul un+2
    U[:,n+2]=U2.reshape(M)                                                      #on ajoute le vecteur à la matrice U
  
    
#----------------------------------------------Affichage de la corde--------------------------------------

x = np.linspace(0, L, M).reshape(M,1)


fig = plt.figure()                                                              # initialise la figure
line, = plt.plot([], []) 
plt.xlim(0, L)
plt.ylim(-2, 2)
plt.xlabel('Longueur')
plt.ylabel('Hauteur')
plt.title("Mouvement de la corde",fontsize=16)

def animate(i):                                                                 #Création de l'animation
    line.set_xdata(x)
    line.set_ydata(U[:,i])
    return line,


#ATTENTION: bien définir le dossier de travail ou le gif ne s'enregistrera pas
ani = animation.FuncAnimation(fig, animate, frames=M+2, blit=True, interval=120)
ani.save('Mouvement de la corde sans contrainte.gif', writer='pillow')          #On sauvegarde en gif




#%% Definir rho

P2min=np.min(np.abs(np.linalg.eigvals(P2)))
P2max=np.max(np.abs(np.linalg.eigvals(P2)))
rho= ((P2min**2)/P2max)


#%% Déduire un+2 avec le Gradient Projeté 

for n in tqdm(range(N)) : 
    b = P1@U[:,n+1] +P0@U[:,n]                                                  #On calcul b
    U2 = Gproj(P2,b,rho,U[:,n],10**(-10)).reshape(M,1)                          #On calcul un+2
    U[:,n+2]=U2.reshape(M)                                                      #On ajoute le vecteur à la matrice U


#------------------------------------------------Affichage de la corde------------------------------------


#x = np.linspace(0, L, M).reshape(M,1)

fig = plt.figure()                                                              #On initialise la figure
line, = plt.plot([],[]) 
plt.xlim(0, L)                                                                  #On définit les axes
plt.ylim(-1, 1)
plt.plot(x,0.5*np.ones(np.shape(x)),c='r')                                      #On affiches des deux barres qui symbolisent les contraintes
plt.plot(x,-0.5*np.ones(np.shape(x)),c='r')
plt.xlabel('Longueur')
plt.ylabel('Hauteur')   
plt.title("Mouvement de la corde",fontsize=16)

def animate(i):                                                                 #Création de l'animation
    line.set_xdata(x)
    line.set_ydata(U[:,i])
    return line,

#ATTENTION: bien définir le dossier de travail ou le gif ne s'enregistrera pas
ani = animation.FuncAnimation(fig, animate, frames=M+2, blit=True, interval=120)
ani.save('Mouvement de la corde avec contraintes.gif', writer='pillow')         #On sauvegarde en .gif





#%% --------------------------------------Testons d'autres conditions initiales------------------------------------

#Saisi des paramètre par l'utilisateur 
print("Veuillez choisir les paramètres de u0(x): \n")
Au = int(input("Amplitude : "))
Bu = int(input("Période : "))

print("\n\nVeuillez choisir les paramètres de v0(x) : \n")
av = int(input("Amplitude : "))
bv = int(input("Période : "))


#calcul de u0 et v0 en fonction des nouveaux paramètres
def u0(x):
   return Au*np.sin(Bu*np.pi*x).reshape(len(x),1) 

def v0(x):
     return -av*np.sin(bv*np.pi*x).reshape(len(x),1)                                   


#Définition de U0 et U1
A=2*np.eye(M)-np.diag(np.ones(M-1),1)-np.diag(np.ones(M-1),-1)
U0 = u0(x)
U1 = U0+dt*v0(x)+0.5*(dt**2)*(np.zeros((len(x),1))-((c/dx)**2)*A@U0)

U=np.zeros((M,N+2))
U[:,0]=U0.reshape(M)
U[:,1]=U1.reshape(M)


fig = plt.figure()                                                              #On initialise la figure
line, = plt.plot([],[]) 
plt.xlim(0, L)                                                                  #On définit les axes
plt.ylim(-1, 1)
plt.plot(x,0.5*np.ones(np.shape(x)),c='r')                                      #On affiches des deux barres qui symbolisent les contraintes
plt.plot(x,-0.5*np.ones(np.shape(x)),c='r')
plt.xlabel('Longueur')
plt.ylabel('Hauteur')   
plt.title("Mouvement de la corde",fontsize=16)

def animate(i):                                                                 #Création de l'animation
    line.set_xdata(x)
    line.set_ydata(U[:,i])
    return line,

#ATTENTION: bien définir le dossier de travail ou le gif ne s'enregistrera pas
ani = animation.FuncAnimation(fig, animate, frames=M+2, blit=True, interval=120)
ani.save('Mouvement de la corde test.gif', writer='pillow')                     #On sauvegarde en .gif



 