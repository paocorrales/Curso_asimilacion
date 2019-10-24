#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:18:10 2019

@author: gimenacasaretto
"""

# =============================================================================
# Filtro de Particulas
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# METODO SIR
# =============================================================================
#
#MODELO KITAGAWA

def modelo_kitagawa(x_0, k, sigma): 
    x = np.zeros(k)
    y = np.zeros(k)
    x[0] = x_0
    y[0] = x_0 + np.random.randn() 
    w = sigma * np.random.randn(k) 
    v = 1 * np.random.randn(k) 
    for it in range(k-1):
        x[it+1] = x[it]*0.5 + 25*x[it]/(1+(x[it]**2)) + 8*(np.cos(1.2*(it-1))) + w[it]
        #y[it+1] = x[it+1] + v[it]
        y[it+1] = x[it+1]**2 /20 + v[it] #operador observacional 
    return x , y 

from Lorenz_63_DA import resample

x_0 = 0 
sigma = 10**0.5
k = 20 

x_true, y = modelo_kitagawa(x_0, k , sigma)

plt.figure()
plt.title ( 'Modelo')
plt.plot(x_true)
plt.plot(y, '.')


Np = 1000 #numero de particulas
pasos= 20
x = np.zeros ((Np , pasos ))
w = np.zeros ((Np , pasos ))
w_norm = np.zeros ((Np , pasos ))

for t in range(pasos-1):
    for j in range(Np):
        x_0 = x[j , t]
        x_aux , basura = modelo_kitagawa(x_0, 2, 10**0.5)
        x[j , t+1] = x_aux[1]
        w[j , t+1] = np.exp(-((y[t+1] - x[j, t+1])**2)/2)
    F = np.sum (w, axis=0)
    w_norm [: , t+1] = w [:, t+1]/ F [t+1]
    indexes = resample(w_norm[:,t+1])
    x[:, t+1] = x [indexes, t+1]
    #w [:, t+1] = 1/Np
plt.figure()
for n in range(pasos-1):
    plt.figure()
    plt.hist(w_norm[:,n] ,bins= 20, alpha = 0.4)
    plt.yscale('log')


x_mean = np.mean(x, axis = 0)

plt.figure()
plt.plot(x_true, label = 'true')
plt.plot(x_mean,label= 'media')
plt.legend()
