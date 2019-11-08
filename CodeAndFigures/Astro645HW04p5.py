#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:04:39 2019

@author: Sandra Bustamante

Irregular orbits: Part 1


"""
import numpy as np
import matplotlib.pyplot as plt
import NumericIntegrations as NI
import SetupPlots as SP
import pandas as pd

#%% Definitions
def totalEnergy(x,v):
    #assuming x=[x1,y1,z1]
    #same for v
    if x.ndim==1:
        K=(1/2)*(v**2).sum()
    else:
        K=(1/2)*(v**2).sum(axis=1)
    U=potential(x)
    E=U+K
    return U,K,E

def potential(X):
    #Two-dimensional non-rotating potential
    #assuming X=[x,y]
    v0=1
    Rc=0.14
    q=0.9
    x=X[:,0]
    y=X[:,1]
    U=(1/2)*v0**2*np.log(Rc**2 + x**2 + (y/q)**2)
    return U

def dvdt(t,x,v):
    dvdt=1 #pendiente
    return dvdt


#%% Common parameters for all orbits
h=1e-3
A=0
B=100 #s

#%% Orbit 1
x10=np.array([2,0])
v10=np.array([0,.401])

IV=np.concatenate((x10.reshape(1,2),v10.reshape(1,2)),axis=0)
t1,x1,v1=NI.leapfrog2D(dvdt,A,B,h,IV)
U1,K1,E1=totalEnergy(x1,v1)

print(E1)

plt.plot(x1[:,0],x1[:,1])