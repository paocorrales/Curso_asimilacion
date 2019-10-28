#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:16:04 2019

@author: pao
"""

# =============================================================================
# Filtro de Kalman por Ensamble
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# METODO ENK
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
        y[it+1] = x[it+1] + v[it]
        #y[it+1] = x[it+1]**2 /20 + v[it] #operador observacional 
    return x , y 

#Utilizando el modelo unidimensional de Kitagawa. Implementar el método EKF y EnKF
#estocástico en este modelo asumiendo que tenemos disponibles observaciones en cada
#paso de tiempo.

#observaciones
pasos = 1000
X_true, obs = modelo_kitagawa(0, pasos, 10**0.5)

plt.figure()
plt.plot(X_true)
plt.plot(obs, '.')

#Kalman     
    
def modelo_tl(x_0):
    x_mtl = 0.5 + 25*(1 - x_0**2)/(1 + x_0**2)**2
    return x_mtl


Pf = np.zeros ((1, pasos))
Pa = np.zeros ((1, pasos))
Q = 10
analisis = np.zeros((1, pasos))
fcst = np.zeros((1, pasos))

R = H = 1
Pf[0, 0] = 1
x_0 = 0

#1D
for i in range(pasos-1):
    #Pronóstico
    fcst[: , i] = x_0*0.5 + 25*x_0/(1+(x_0**2)) + 8*(np.cos(1.2*(i-1))) 
    
    #Análisis
    K = Pf[:, i]*H*(R + H*Pf[:,i]*H)**-1
    analisis[:, i] = fcst[:, i] + K*(obs[i]- fcst[:, i])
         
    Pa[:, i] = (np.eye(1) - K*H)*Pf[:, i]
    Pf[:, i+1] = modelo_tl(x_0)*Pa[:, i]*modelo_tl(x_0) + Q

    x_0 = analisis[:, i]

plt.figure()
plt.title("EKF")
plt.plot(analisis[0,:], label ='analisis')
plt.plot(X_true, '--', label = 'verdad')
plt.plot(fcst[0,:], label='fcst' )
plt.plot(obs, ".", label = 'obs')
plt.legend()

plt.figure()
plt.plot(X_true - analisis[0,:])
plt.plot(X_true - fcst[0,:])
plt.plot(analisis[0,:] - fcst[0,:])


plt.figure()
plt.plot(Pf[0,:], np.abs(X_true - fcst[0,:]),'*')
#plt.scatter(Pf[0,:], np.abs(X_true - fcst[0,:]))

#for i in range(pasos-1):
#    
#    K = Pf[:, i].dot(H.T).dot(np.linalg.inv(R + H.dot(Pf[:,i]).dot(H.T)))
#    
#    analisis[:, i] = fcst[:, i] + K.dot((obs[i]- fcst[:, i]))
#    
#    x_0 = analisis[:, i]
#    x_aux , basura = modelo_kitagawa(x_0, 2, 10**0.5)
#    fcst[: , i+1] = x_aux[1]
#    
#    Pa [:, i] = (np.eye(1) - K.dot(H)).dot(Pf[:, i])
#    
#    Pf[:, i+1] = modelo_tl(x_0).dot(Pa[:, i]).dot(modelo_tl(x_0))
    
