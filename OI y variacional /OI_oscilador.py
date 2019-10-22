#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:06:30 2019

@author: gimenacasaretto
"""

#Interpolacion optima modelo lineal 
#Considerar el modelo del oscilador armónico 
#(asumir un dt=0.1 y un Ω^2​ ​=2.0 y una condición inicial con x=0 m y v=1 m/s). 
#Generar una simulación de 200 pasos de tiempo que represente 
#la verdadera evolución del sistema. Generar un conjunto de observaciones 
#de la posición y la velocidad a partir de la verdad asumiendo un error 
#observacional Gaussiano aditivo con desviación estándar q de 1 ​m para la 
#posición y 1 ​m/s​ para la velocidad.

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
#  1 (a)
# =============================================================================
#OI : xa = xb + K(y - Hxb) yo quiero mi xa
#H en este caso es la identidad, 
#porque estoy observando en el mismo punto el estado 
#B tambien es la identidad por enunciado
#R como asumo que no hay correlacion entre las observaciones 
#entonces es la identidad
#K = B*H.T ( B + H.T*R*H)^-1

xb = 2
vb = 1

analisis = np.zeros ((200, 2))

obs = np.zeros ((200, 2))
obs[:,0] = obs_x
obs[:,1] = obs_v

fcst= np.zeros ((200, 2))
fcst [0,0] = xb
fcst [0,1] = vb 

R = H = B = np.array ([[1, 0 ], [0 , 1]]) 
B = 0.1*B
K = B.dot(H.T).dot(np.linalg.inv(B + H.T.dot(R).dot(H)))

for i in range(len(analisis)-1):
    analisis[i , :] = fcst[i, : ] + K.dot((obs[i,:]- fcst[i,:]))
    fcst [i + 1 , 1 ]= analisis [i, 1 ] - dt*omega*analisis[i,0]
    fcst [i + 1, 0]= analisis [i, 0] + dt*fcst[i+1,1]


plt.figure()
plt.subplot(2,1,1)
plt.title ( 'Posición')
plt.plot(x_true, label= 'posición ')
plt.plot(analisis[:,0], label= 'posición modelo')
plt.plot(obs_x, '.', label= 'obs x', alpha= 0.5)
plt.legend()

plt.subplot(2,1,2)
plt.title('Velocidad')
plt.plot(v_true, label= 'velocidad')
plt.plot(analisis[:,1], label= 'velocidad modelo')
plt.plot(obs_v, '.', label = 'obs v', alpha= 0.5)
plt.legend()
plt.tight_layout()
plt.savefig( 'OI oscilador B 0.1.png', dpi=500)

# =============================================================================
# 1 (b)
# =============================================================================
# B = [var(x), cov(x,y)] [cov(x,y), var(y)]

# =============================================================================
# 1 (b) bis 
# =============================================================================

x_dif = x_true - analisis [:,0]
v_dif = v_true - analisis [:,1]

plt.figure()
plt.plot(x_dif)
plt.plot(v_dif)

#al estar suponiendo que la estimacion tienen
#siempre la misma incertidumbre no varia la confianza
#en el tiempo. Por que es lineal 
#el P no se achica con el tiempo , es cte con el tiempo 
#es lo que hace que no le damos peso al background 
#entonces el error nunca se achica mas alla de cierto valor . 

# =============================================================================
# 1 (c) NO es realista.
# =============================================================================

# =============================================================================
# 1 (d) Modelo imperfecto 
# =============================================================================
v_true= np.zeros(200)
x_true= np.zeros(200)
x_true[0]=0 
v_true[0]=1

for k in range(len(v_true)-1):
    v_true[k+1]= v_true[k] - dt*omega*x_true[k]
    x_true[k+1]= x_true[k] +dt*v_true[k+1]
    
error_x = np.random.randn(len(x_true))
error_v = np.random.randn(len(v_true))

x_true = x_true + error_x
v_true = v_true + error_v


plt.figure()
plt.plot(obs_x, '.', label= 'obs x')
plt.plot(obs_v, '.', label = 'obs v')
plt.plot(x_true, label= 'posición')
plt.plot(v_true, label= 'velocidad')
plt.legend()
plt.savefig( 'Oscilador 1(d).png', dpi=500)

xb = 2
vb = 1

analisis = np.zeros ((200, 2))
analisis [0,0] = xb
analisis [0,1] = vb
obs = np.zeros ((200, 2))
obs[:,0] = obs_x
obs[:,1] = obs_v

K= np.array ([[2, 0 ], [0 , 2]]) 
K = np.linalg.inv(K)

for i in range(len(analisis)-1):
    analisis[i , :] = fcst[i, : ] + K.dot((obs[i,:]- fcst[i,:]))
    fcst [i + 1 , 1 ]= analisis [i, 1 ] - dt*omega*analisis[i,0]
    fcst [i + 1, 0]= analisis [i, 0] + dt*fcst[i+1,1]

#Habria que modificar la B ya que no tengo un modelo perfecto 
#podria ver con distintos valores para ver cual reduce el error
#da una mejor estimación 

plt.figure()
plt.subplot(2,1,1)
plt.title ( 'Posición')
plt.plot(x_true, label= 'posición ')
plt.plot(analisis[:,0], label= 'posición modelo')
plt.plot(obs_x, '.', label= 'obs x', alpha= 0.5)
#plt.legend()

plt.subplot(2,1,2)
plt.title('Velocidad')
plt.plot(v_true, label= 'velocidad')
plt.plot(analisis[:,1], label= 'velocidad modelo')
plt.plot(obs_v, '.', label = 'obs v', alpha= 0.5)
#plt.legend()
plt.tight_layout()
plt.savefig( 'OI oscilador 1(d).png', dpi=500)