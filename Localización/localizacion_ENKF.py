#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:22:13 2019

@author: pao
"""
import numpy as np
import matplotlib.pyplot as plt

L = 4
EnSize = 100
B = np.zeros((10,10))

for i in range(10):
    for j in range(10):
        B[i, j] = np.exp(- (i - j)**2/(2*L))
        
Pe = np.zeros((10,10,100))
for i in range(100):
    fcst = np.random.multivariate_normal(np.zeros(10), B, 10000)
    Pe[:,:,i] = np.cov(fcst.T)

var_Pe = np.var(Pe, axis = 2)

plt.figure()
plt.imshow(B)
plt.colorbar()

plt.figure()
plt.imshow(Pe[:,:,9])
plt.colorbar()

plt.figure()
plt.imshow((var_Pe**0.5)/B)
plt.colorbar()


# Filtro de Kalman

L = 4

B = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        B[i, j] = np.exp(- (i - j)**2/(2*L))
        
X_true = np.random.multivariate_normal(np.zeros(10), B, 1).T

obs = X_true[4] + 2*np.random.randn(1)

EnSize = 100
pasos = 1000

H = np.zeros((1, 10))
H[0, 4] = 1

R = 2
    
fcst = np.zeros((EnSize, 10, pasos))
ana = np.zeros((EnSize, 10, pasos)) 
Pe = np.zeros((10,10,pasos))
Pa = np.zeros((10,10,pasos))
estimacion = np.zeros((EnSize, 10, pasos))
  
for p in range(pasos):
    X_true = np.random.multivariate_normal(np.zeros(10), B, 1).T
    obs = X_true[4] + 2*np.random.randn(1)
    true = np.repeat(X_true, 100, axis = 1).T 
    
    fcst[:,:,p] = np.random.multivariate_normal(np.zeros(10), B, EnSize)

    Pe[:,:,p] = np.cov(fcst[:,:,p].T)
    
    K = Pe[:,:,p].dot(H.T).dot(np.linalg.inv(H.dot(Pe[:,:,p]).dot(H.T) + R))
    
    for e in range(EnSize):

        ana[e,:,p] = fcst[e,:,p] + (K*(obs - H.dot(fcst[e,:,p].T))).T

    Pa[:,:,p] = (np.eye(10) - K.dot(H)).dot(Pe[:,:,p])
    
    estimacion[:,:,p] = true - ana[:,:,p]
    
var_Pe = np.var(Pe, axis = 2)
var_Pa = np.var(Pa, axis = 2)
var_estimacion = np.var(estimacion, axis = 2)

mean_Pe = np.mean(Pe, axis = 2)
mean_Pa = np.mean(Pa, axis = 2)
mean_estimacion = np.mean(estimacion, axis = 2)

plt.figure()
plt.subplot(2,2,1)
plt.title("var_Pa")
plt.imshow(var_Pa)
plt.colorbar()
plt.subplot(2,2,2)
plt.title("var_Pe")
plt.imshow(var_Pe)
plt.colorbar()
plt.subplot(2,2,3)
plt.title("mean_Pa")
plt.imshow(mean_Pa)
plt.colorbar()
plt.subplot(2,2,4)
plt.title("mean_Pe")
plt.imshow(mean_Pe)
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.plot(mean_estimacion.T)
plt.plot(np.mean(mean_estimacion, axis = 0), 'k')
