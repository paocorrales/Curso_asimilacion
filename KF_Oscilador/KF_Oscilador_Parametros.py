#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:18:54 2019

@author: pao
"""

# =============================================================================
# Método Kalman Filter Lineal en el oscilador armónico
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

dt= 0.1
omega= 2

v_true= np.zeros(200)
x_true= np.zeros(200)
x_true[0]=0 
v_true[0]=1

for k in range(len(v_true)-1):
    v_true[k+1]= v_true[k] - dt*omega*x_true[k]
    x_true[k+1]= x_true[k] +dt*v_true[k+1]
    
error_x = np.random.randn(len(x_true))
error_v = np.random.randn(len(v_true))

obs_x = x_true + error_x
obs_v = v_true + error_v

plt.figure()
plt.plot(obs_x, 'o', label= 'obs x')
plt.plot(obs_v, '*', label = 'obs v')
plt.plot(x_true, label= 'posición')
plt.plot(v_true, label= 'velocidad')
plt.legend()
plt.savefig( 'Oscilador.png', dpi=500)

# =============================================================================
# Partiendo de una estimación inicial del estado del sistema con x=2 ​ m y v=1 ​ m/s ​ ,
# realizar una estimación secuencial del estado del sistema utilizando el método de
# filtro de Kalman. Asumir una matriz P inicial diagonal con una desviación estándar
# de los errores en posición y en velocidad de 1 m y 1 m/s respectivamente.
# 
# Analizar la evolución de B
# =============================================================================
pasos = 20
X = np.zeros((2, pasos))
X[0, 0] = 2 
X[1, 0] = 1

dt= 0.1
omega= 2
M = np.array([[1 - (dt*omega)**2, dt], [-dt*omega**2, 1]])

B = np.zeros ((2, 2, pasos))
B[:,:,0] = np.array ([[1, 0 ], [0 , 1]]) 

Q = np.random.randn(2, 2)

for k in range(pasos-1):
    X[1, k+1]= X[1, k] - dt*omega*X[0, k]
    X[0, k+1]= X[0, k] + dt*X[1, k+1]
    
    B[:, :, k+1] = M.dot(B[:,:,k]).dot(M.T) 
    
plt.figure()
plt.title("Evolución de la matriz B")
plt.plot (B [0,0,:], label = "var x")
plt.plot (B [0,1,:], label = "cov(x,v)")
plt.plot (B [1,0,:], "*", label = "cov(v,x)")
plt.plot (B [1,1,:], label = "var v")
plt.legend()


# =============================================================================
# Asimile las observaciones generadas durante los 200 pasos de tiempo. Estudiar
# la convergencia de la estimación al verdadero estado del sistema. Analice la
# evolución de la matriz de covarianza en este caso. Discuta las diferencias con lo
# encontrado en el punto anterior.
# =============================================================================
pasos = 1000
dt= 0.1

def oscilador(X_0, omega, dt = 0.1):
    M = np.array([[1 - (dt*omega)**2, dt], [-dt*omega**2, 1]])
    X = M.dot(X_0)
    
    return X
    

X_true = np.zeros((2, pasos))
X_true[0, 0] = 1 
X_true[1, 0] = 1

for k in range(pasos-1):
    X_true[:, k+1]= oscilador(X_true[:, k], omega = 2)
    
obs = X_true + np.random.randn(2, pasos)
    
B = np.zeros ((3, 3, pasos))
B[:,:,0] = np.eye(3)*1
B[2,2,0] = 0.5

Pa = np.zeros ((3, 3, pasos))

ensize = 50
analisis = np.zeros ((3, pasos, ensize))

fcst= np.zeros ((3, pasos, ensize))

R = np.eye(2) 
H = np.array ([[1, 0, 0], [0 , 1, 0]]) 

fcst [0,0,:] = 2 + np.random.randn(ensize).T
fcst [1,0,:] = 1 + np.random.randn(ensize).T
fcst [2,0,:] = 1.5 + 0.1*np.random.randn(ensize).T
        
for i in range(pasos-1):
    
    for e in range(ensize):
        
        K = B[:,:,i].dot(H.T).dot(np.linalg.inv(R + H.dot(B[:,:,i]).dot(H.T)))
        
        analisis[:, i, e] = fcst[:, i, e] + K.dot((obs[:, i] + np.random.randn(2).T - fcst[0:2, i, e]))
        
        fcst[0:2,i+1,e] = oscilador(analisis[0:2,i,e], analisis[2,i,e])
        fcst[2,i+1,e] = analisis[2,i,e]
        
#    Pa [:,:,i] = np.cov(analisis[:,i,:], analisis[:,i,:])
        
    B[:,:,i+1] = np.cov(fcst[:,i+1,:])

plt.figure()
plt.title("Omega")
plt.axhline(y = 2, color = 'k', linestyle = '--')
plt.plot(analisis[2,0:-1,:])
    
    
plt.figure()
plt.title("Posición")
plt.plot(analisis[0,:,:])
plt.plot(X_true[0,:], '--')
plt.plot(obs[0,:], ".")
plt.figure()
plt.title("Velocidad")
plt.plot(analisis[1,:])
plt.plot(X_true[1,:], '--')
plt.plot(obs[1,:], ".")



plt.figure()
plt.plot(analisis[0,:] - X_true[0,:], label = "x")
plt.plot(analisis[1,:] - X_true[1,:], label = "v")
plt.legend()

plt.figure()
plt.title("Evolución de la matriz Pa")
plt.plot (Pa [0,0,:], label = "var x")
plt.plot (Pa [0,1,:], label = "cov(x,v)")
plt.plot (Pa [1,0,:], "*", label = "cov(v,x)")
plt.plot (Pa [1,1,:], label = "var v")
plt.legend()
