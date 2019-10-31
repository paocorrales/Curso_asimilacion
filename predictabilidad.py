#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:55:20 2019

@author: pao
"""

import numpy as np
import matplotlib.pyplot as plt

import Lorenz_63 as model
import Lorenz_63_DA as da
import pandas as pd

a = 10.0
r = 28.0
b = 8.0/3.0
p = np.array([a, r, b])

dt = 0.1
numstep = 1000
x0 =  np.array([8.0, 0.0, 30.0])
numtrans = 600

bst = 8       #Cada cuantos pasos guardo la solución
forecast_length = 50

EnsSize = 1000
nvars = 3

pert_amp = 1.0

ini_time = 100

x = np.copy(x0)

for i in range(numtrans) :
    x = model.forward_model(x, p, dt)
    
state =  np.zeros((numstep, nvars))

for i in range(numstep) :
    for j in range(bst) :
        x = model.forward_model(x, p, dt)
    state[i, :] = x
    
plt.figure()
plt.plot(state[0:500, 0])
plt.plot(state[0:500, 1])
    
df = pd.DataFrame(state)

df.to_csv("/home/pao/curso_asimilacion/lorenz_63.csv")


sigma_x = np.random.multivariate_normal(np.zeros(3), np.identity(3), 1000)

new_x0 = state[199, :] + sigma_x
state_new =  np.zeros((50, nvars, 1000))
for k in range(1000) :
    x = new_x0[k, :]
    for i in range(50) :
        for j in range(bst) :
            x = model.forward_model(x, p, dt)
        state_new[i, :, k] = x


plt.figure()
for i in range(1000): 
    plt.plot(state_new[:, 0, i], 'k')
