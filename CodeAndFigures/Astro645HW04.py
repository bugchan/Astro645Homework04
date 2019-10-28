#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:40:25 2019

@author: Sandra Bustamante
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import NumericIntegrations as NI
import SetupPlots as SP

#%% Definitions
def dvdt(t,x,v):
  #Pendulum equation of motion
  dvdt=-constants.g*np.sin(x)
  return dvdt

def energy(v,x):
  #Energy of pendulum
  E=0.5*v**2+constants.g*(1-np.cos(x))
  return E

#%% Pendulum

e=1e-2
Ecrit=2*constants.g #
#rotation,libration, near unstable equilibrium
t0=np.array([0,0,0])
x0=np.array([-1.3*np.pi,.5,np.pi-e])
v0=np.array([.4,0,0])*constants.g

T=1/np.sqrt(constants.g) #Period

N=100000
a=0
b=T*10000
IV=np.concatenate((x0.reshape(3,1),v0.reshape(3,1)),
                  axis=1)

#%%Rotation

tRot,xRot,vRot=NI.leapfrog(dvdt,a,b,N,IV[0])
tRotRK,xRotRK,vRotRK=NI.RK4(dvdt,a,b,N,IV[0])
ERot=energy(vRot,xRot)
ERotRK=energy(vRotRK,xRotRK)
errRot=ERot-ERotRK

#%%Libration
tLib,xLib,vLib=NI.leapfrog(dvdt,a,b,N,IV[1])
tLibRK,xLibRK,vLibRK=NI.RK4(dvdt,a,b,N,IV[1])
ELib=energy(vLib, xLib)
ELibRK=energy(vLibRK, xLibRK)
errLib=ELib-ELibRK

#%%Near unstable position
tUns, xUns,vUns=NI.leapfrog(dvdt,a,b,N,IV[2])
tUnsRK,xUnsRK,vUnsRK=NI.RK4(dvdt,a,b,N,IV[2])
EUns=energy(vUns,xUns)
EUnsRK=energy(vUnsRK,xUnsRK)
errUns=EUns-EUnsRK


#%% Plot Phase Space Curve
width,height=SP.setupPlot(singleColumn=True)
fig1 = plt.figure(figsize=(width,.5*height))
grid = plt.GridSpec(1,2)

#Leapfrog
ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(xRot,vRot,'--',label='Rotation')
ax1.plot(xLib,vLib,'-.',label='Libration')
ax1.plot(xUns,vUns,':',label='Near Unstable Equilibrium')
ax1.set_xlabel(r'Angle $\theta$ [rad]')
ax1.set_xlim(-1.2*np.pi,1.2*np.pi)
ax1.set_ylabel(r'Velocity $v$ [m/s]')
ax1.set_title('Leapfrog Integration')
#ax1.set_aspect('equal')
ax1.grid()
#ax1.legend()

#RK4
ax4 = fig1.add_subplot(grid[0,1],sharey=ax1)
ax4.plot(xRotRK,vRotRK,'--',label='Rotation')
ax4.plot(xLibRK,vLibRK,'-.',label='Libration')
ax4.plot(xUnsRK,vUnsRK,':',label='Near Unstable Equilibrium')
ax4.set_xlabel(r'Angle $\theta$ [rad]')
ax4.set_xlim(-1.2*np.pi,1.2*np.pi)
#ax4.set_ylabel(r'Velocity $v$ [m/s]')
#ax4.set_ylim([])
ax4.set_title('Runge-Kutta Integration')
ax4.grid()
ax4.legend()

#%% Plot Energy vs time
#width,height=SP.setupPlot(singleColumn=True)
fig2 = plt.figure(figsize=(width,height))
grid1 = plt.GridSpec(1,1)

ax2 = fig2.add_subplot(grid1[0,0])
ax2.plot(tRot,ERot,'--',label='Rotation')
#ax2.plot(tRotRK,ERotRK,'--',label='RotationRK')
#ax2.plot(tLib,ELib,'-.',label='Libration')
#ax2.plot(tUns,EUns,':',label='Near Unstable Equilibrium')
#ax2.hlines(Ecrit,a,b,label='Critical Energy')
ax2.set_xlabel('Time [s]')
ax2.set_xlim(a,b/(.01*N))
ax2.set_ylabel('Total Energy $E_T$ []')
ax2.grid()
ax2.legend()

#%%  Phase Space Plot of Unstable equilibrium
width,height=SP.setupPlot(singleColumn=True)
fig3 = plt.figure(figsize=(width,height))
#grid1 = plt.GridSpec(1,1)

ax3 = fig3.add_subplot(grid1[0,0])
ax3.plot(xUns,vUns,':',label='Leapfrog')
ax3.plot(xUnsRK,vUnsRK,':',label='RK4')
ax3.set_title('Unstable Equilibrium')
ax3.set_xlim(-1.1*np.pi,1.1*np.pi)
ax3.grid()
ax3.legend()

#%% Plot comparing Leapfrog and RK4
width,height=SP.setupPlot(singleColumn=True)
fig4 = plt.figure(figsize=(width,.5*height))
gridf4 = plt.GridSpec(1,3)

#Near Unstable Equilbrium
ax4 = fig4.add_subplot(gridf4[0,2])
ax4.plot(tUns,EUns,':',label='Leapfrog')
ax4.plot(tUnsRK,EUnsRK,':',label='RK4')
ax4.set_xlim(a,b)
ax4.set_xlabel('Time [s]')
#ax4.set_ylabel('Total Energy $E_T$ []')
ax4.set_title('Near Unstable Equilibrium')
ax4.grid()
ax4.legend(loc='upper left')

#Libration
ax5 = fig4.add_subplot(gridf4[0,1])
ax5.plot(tLib,ELib,':',label='Leapfrog')
ax5.plot(tLibRK,ELibRK,':',label='RK4')
ax5.set_xlim(a,b)
ax5.set_xlabel('Time [s]')
#ax5.set_ylabel('Total Energy $E_T$ []')
ax5.set_title('Libration')
ax5.grid()
ax5.legend(loc='upper left')

#Rotation
ax6 = fig4.add_subplot(gridf4[0,0])
ax6.plot(tRot,ERot,':',label='Leapfrog')
ax6.plot(tRotRK,ERotRK,':',label='RK4')
ax6.set_xlim(a,b)
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('Total Energy $E_T$ []')
ax6.set_title('Rotation')
ax6.grid()
ax6.legend(loc='upper left')

fig4.tight_layout()

#%% Error plot
width,height=SP.setupPlot(singleColumn=False)
fig5 = plt.figure(figsize=(width,height))
gridf5 = plt.GridSpec(1,1)

ax7 = fig5.add_subplot(gridf5[0,0])
ax7.plot(errLib)
