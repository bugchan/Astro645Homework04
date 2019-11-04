#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:07:03 2019

@author: sbustamanteg
"""

import numpy as np
import matplotlib.pyplot as plt
import NumericIntegrations as NI
import SetupPlots as SP

def potential(X):
    #assuming X=[x,y]
    a=1/2
    x=X[:,0]
    y=X[:,1]
    U=-((x-a)**2+y**2)**(-1/2)-((x+a)**2+y**2)**(-1/2)
    return U

def dvdt(x,v,t):
    #assuming X=[x,y]
    #Non-axisymmetric orbits with 2 point masses
    a=1/2
    P1=(x[0]-a+x[1])*((x[0]-a)**2+x[1]**2)**(-3/2)
    P2=(x[0]+a+x[1])*((x[0]-a)**2+x[1]**2)**(-3/2)
    dvdt=P1+P2
    return dvdt

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

#%% Common parameters for all orbits
h=1e-3
A=0
B=100 #s

a=1/2

#%%
x0=np.array([.01,0])
v0=np.array([0,.01])

IV=np.concatenate((x0.reshape(1,2),v0.reshape(1,2)),axis=0)
t1,x1,v1=NI.leapfrog2D(dvdt,A,B,h,IV)
U,K,E=totalEnergy(x1,v1)

#%% plot x-y
width,height=SP.setupPlot(singleColumn=True)
fig1 = plt.figure(figsize=(width,height))
grid = plt.GridSpec(1,1)

ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(x1[:,0],x1[:,1])

#%% plot phase space
width,height=SP.setupPlot(singleColumn=True)
fig2 = plt.figure(figsize=(width,height))
grid = plt.GridSpec(1,2)

ax2 = fig2.add_subplot(grid[0,0])
ax2.plot(x1[:,0],v1[:,0])

ax3 = fig2.add_subplot(grid[0,1])
ax3.plot(x1[:,1],v1[:,1])

#%% plot UKE
width,height=SP.setupPlot(singleColumn=True)
fig3 = plt.figure(figsize=(width,height))
grid = plt.GridSpec(1,2)

ax3 = fig3.add_subplot(grid[0,0])
ax3.plot(U,label='U')
ax3.plot(K,label='K')
ax3.plot(E,label='E')
ax3.legend()
