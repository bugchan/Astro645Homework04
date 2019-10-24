#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:40:25 2019

@author: Sandra Bustamante
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

#%% Definitions
def setupPlot(singleColumn):
  
  if singleColumn:
    fontsize=10
    width=6.9
    linewidth=1
  else:
    fontsize=8
    width=3.39
    linewidth=0.8
    
  height=width*(np.sqrt(5.)-1.)/2.  
  params = {'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'font.size': fontsize,
            'legend.fontsize': fontsize-2,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'lines.linewidth': linewidth,
            'grid.linewidth' : linewidth*.8,
            'axes.axisbelow' : True
            }
  plt.rcParams.update(params)
  return width,height

def dvdt(x):
  a=-constants.g*np.sin(x)
  return a

def leapfrog(t,x0,v0,N):
  h=t[1]-t[0]
  x=np.zeros(N)
  v=np.zeros(N)
  v[0]=v0
  x[0]=x0
  for n in np.arange(1,N):
    k1=x[n-1]+0.5*h*v[n-1]
    v[n]=v[n-1]+h*dvdt(k1)
    x[n]=k1+0.5*h*v[n]
  return x,v
  

#%% Pendulum

e=1e-2
#rotation,libration, near unstable equilibrium
Ecrit=2*constants.g # 
x0=[np.pi,.5,np.pi-e]

v0=0
N=10000
t=np.linspace(0,30,N)

#Rotation
xRot,vRot=leapfrog(t,x0[0],v0,N)
ERot=0.5*vRot**2 - np.cos(xRot)
#Libration
xLib,vLib=leapfrog(t,x0[1],v0,N)
ELib=0.5*vLib**2 - np.cos(xLib)
#Near unstable position
xUns,vUns=leapfrog(t,x0[2],v0,N)
EUns=0.5*vUns**2 - np.cos(xUns)

#%% Plot Phase Space Curve
width,height=setupPlot(singleColumn=True)
fig1 = plt.figure(figsize=(width,height))
grid = plt.GridSpec(1,1)

ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(xRot,vRot,'--',label='Rotation')
ax1.plot(xLib,vLib,'-.',label='Libration')
ax1.plot(xUns,vUns,':',label='Near Unstable Equilibrium')
ax1.set_xlabel('Position [x]')
ax1.set_ylabel('Velocity [v]')
ax1.grid()
ax1.legend()

#%% Plot Energy
#width,height=setupPlot(singleColumn=True)
fig2 = plt.figure(figsize=(width,height))
grid = plt.GridSpec(1,1)

ax2 = fig2.add_subplot(grid[0,0])
ax2.plot(t,ERot,'--',label='Rotation')
ax2.plot(t,ELib,'-.',label='Libration')
ax2.plot(t,EUns,':',label='Near Unstable Equilibrium')
ax2.hlines(Ecrit,t.min(),t.max(),label='Critical Energy')
ax2.set_xlabel('Time [t]')
ax2.set_ylabel('Total Energy [E]')
ax2.grid()
ax2.legend()