#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:06:39 2019

@author: pao
"""


# =============================================================================
# Filtro de Particulas
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

import Lorenz_63 as model
import Lorenz_63_DA as da

# =============================================================================
# METODO SIR
# =============================================================================
#
# Lorenz 63

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