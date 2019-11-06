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
  dvdt=-np.sin(x)
  #print(dvdt)
  return dvdt

def energy(v,x):
  #Energy of pendulum
  E=(1/2)*v**2+(1-np.cos(x))
  return E

#%% Pendulum

e=1e-2
Ecrit=2 #
#rotation,libration, near unstable equilibrium
x0=np.array([-1.3*np.pi,.5,np.pi-e])
v0=np.array([1.3,0,0])

T=1 #Period

a=0
b=T*10000
h=0.1
IV=np.concatenate((x0.reshape(3,1),v0.reshape(3,1)),
                  axis=1)

#%%Rotation
print('Starting calculating pendulum in rotation')
print('Doing Leapfrog')
tRot,xRot,vRot=NI.leapfrog2D(dvdt,a,b,h,IV[0],dim=1)
print('Doing RK4')
tRotRK,xRotRK,vRotRK=NI.RK4(dvdt,a,b,h,IV[0],dim=1)
ERot=energy(vRot,xRot)
ERotRK=energy(vRotRK,xRotRK)
errRot=ERot-ERotRK

plt.plot(xRot,vRot,'--',label='Rotation')
print('Energy:',ERot)

#%%Libration
print('Starting calculating pendulum in libration')
print('Doing Leapfrog')
tLib,xLib,vLib=NI.leapfrog2D(dvdt,a,b,h,IV[1],dim=1)
print('Doing RK4')
tLibRK,xLibRK,vLibRK=NI.RK4(dvdt,a,b,h,IV[1],dim=1)
ELib=energy(vLib, xLib)
ELibRK=energy(vLibRK, xLibRK)
errLib=ELib-ELibRK

#%%Near unstable position
print('Starting calculating pendulum near unstable position')
print('Doing Leapfrog')
tUns, xUns,vUns=NI.leapfrog2D(dvdt,a,b,h,IV[2],dim=1)
print('Doing RK4')
tUnsRK,xUnsRK,vUnsRK=NI.RK4(dvdt,a,b,h,IV[2],dim=1)
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
ax1.plot(xUns,vUns,':',label='Near Unstable Eq')
ax1.set_xlabel(r'Angle $\theta$ [rad]')
ax1.set_xlim(-1.2*np.pi,1.2*np.pi)
ax1.set_ylabel(r'Velocity $v$ [m/s]')
ax1.set_title('Leapfrog Integration')
ax1.grid()

#RK4
ax4 = fig1.add_subplot(grid[0,1],sharey=ax1)
ax4.plot(xRotRK,vRotRK,'--',label='Rotation')
ax4.plot(xLibRK,vLibRK,'-.',label='Libration')
ax4.plot(xUnsRK,vUnsRK,':',label='Near Unstable Eq')
ax4.set_xlabel(r'Angle $\theta$ [rad]')
ax4.set_xlim(-1.2*np.pi,1.2*np.pi)
ax4.set_title('Runge-Kutta Integration')
ax4.grid()
ax4.legend(bbox_to_anchor=(1.1, 0.6))

fig1.tight_layout()
fig1.savefig('PendulumPhaseSpace.pdf')

#%% Plot Energy vs time
#width,height=SP.setupPlot(singleColumn=True)
fig2 = plt.figure(figsize=(width,height))
grid1 = plt.GridSpec(1,1)

ax2 = fig2.add_subplot(grid1[0,0])
ax2.plot(tRot,ERot,'--',label='Rotation Leapfrog')
ax2.plot(tRotRK,ERotRK,'--',label='RotationRK')
#ax2.plot(tLib,ELib,'-.',label='Libration')
#ax2.plot(tUns,EUns,':',label='Near Unstable Equilibrium')
#ax2.hlines(Ecrit,a,b,label='Critical Energy')
ax2.set_xlabel('Time [s]')
ax2.set_xlim(a,b)
ax2.set_ylabel('Total Energy $E_T$ []')
ax2.grid()
ax2.legend(loc='lower right')

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
ax3.legend(loc='lower right')

#%% Plot comparing Leapfrog and RK4
width,height=SP.setupPlot(singleColumn=True)
fig4 = plt.figure(figsize=(width,.5*height))
gridf4 = plt.GridSpec(1,3)

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
#ax6.set_ylim(2.43275,2.4329)

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
#ax5.set_ylim(0,.001)

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

fig4.tight_layout()

#%% Error plot
width,height=SP.setupPlot(singleColumn=False)
fig5 = plt.figure(figsize=(width,height))
gridf5 = plt.GridSpec(1,1)

ax7 = fig5.add_subplot(gridf5[0,0])
ax7.plot(errLib)
