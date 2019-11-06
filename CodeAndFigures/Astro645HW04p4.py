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
import pandas as pd

#%% Definitions
def potential(X):
    #assuming X=[x,y]
    a=1/2
    x=X[:,0]
    y=X[:,1]
    U=-((x-a)**2+y**2)**(-1/2)-((x+a)**2+y**2)**(-1/2)
    return U

def dvdt(t,x,v):
    #assuming X=[x,y]
    #Non-axisymmetric orbits with 2 point masses
    a=1/2
    eps=1e-4
    N1=((x[0]-a)**2+x[1]**2+eps**2)**(-1.5)
    N2=((x[0]+a)**2+x[1]**2+eps**2)**(-1.5)
    dvxdt=-(x[0]-a)*N1 - (x[0]+a)*N2
    dvydt=-x[1]*N1 - x[1]*N2
    dvdt=np.array([dvxdt,dvydt])
    #print(x,v,dvdt)
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

#%% Orbit 1
x10=np.array([2,0])
v10=np.array([0,.4])

IV=np.concatenate((x10.reshape(1,2),v10.reshape(1,2)),axis=0)
t1,x1,v1=NI.leapfrog2D(dvdt,A,B,h,IV)
U1,K1,E1=totalEnergy(x1,v1)

#%% Orbit 2
x20=np.array([2,0])
v20=np.array([0,0.5])

IV=np.concatenate((x20.reshape(1,2),v20.reshape(1,2)),axis=0)
t2,x2,v2=NI.leapfrog2D(dvdt,A,B,h,IV)
U2,K2,E2=totalEnergy(x2,v2)

#%% Orbit 3
#x30=np.array([3,0])
#v30=np.array([0,0.6])
x30=np.array([0,1])
v30=np.array([0.9,0])
#x30=np.array([0,1.1])
#v30=np.array([.01,0.301])

IV=np.concatenate((x30.reshape(1,2),
                   v30.reshape(1,2)),axis=0)
t3,x3,v3=NI.leapfrog2D(dvdt,A,B,h,IV)
U3,K3,E3=totalEnergy(x3,v3)
print(E3)

#%% plot x-y
width,height=SP.setupPlot(singleColumn=True)
fig1 = plt.figure(figsize=(.9*width,.5*height))
grid = plt.GridSpec(1,3)

ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(x1[:,0],x1[:,1], label='Orbit 0')
ax1.grid()
ax1.set_xlabel('x-position')
ax1.set_ylabel('y-position')
ax1.legend(loc='lower right')

ax2 = fig1.add_subplot(grid[0,1],sharey=ax1)
ax2.plot(x2[:,0],x2[:,1],label='Orbit 1')
ax2.grid()
ax2.set_xlabel('x-position')
ax2.legend(loc='lower right')

ax3 = fig1.add_subplot(grid[0,2])
ax3.plot(x3[:,0],x3[:,1],label='Orbit 2')
ax3.grid()
ax3.set_xlabel('x-position')
ax3.legend(loc='lower right')

fig1.tight_layout()
fig1.savefig('NonAxisSymetricOrbits.pdf')

#%% plot phase space
width,height=SP.setupPlot(singleColumn=True)
fig2 = plt.figure(figsize=(.9*width,1*height))
grid = plt.GridSpec(2,3)

ax4 = fig2.add_subplot(grid[0,0])
ax4.plot(x1[:,0],v1[:,0],label='Orbit 0')
ax4.grid()
ax4.legend(loc='lower right')
ax4.set_ylabel(r'$v_x$')
ax4.set_xlabel(r'$x$')

ax5 = fig2.add_subplot(grid[1,0])
ax5.plot(x1[:,1],v1[:,1],label='Orbit 0')
ax5.grid()
ax5.legend(loc='lower right')
ax5.set_ylabel(r'$v_y$')
ax5.set_xlabel(r'$y$')

ax6 = fig2.add_subplot(grid[0,1])
ax6.plot(x2[:,0],v2[:,0],label='Orbit 1')
ax6.grid()
ax6.legend(loc='lower right')
ax6.set_xlabel(r'$x$')

ax7 = fig2.add_subplot(grid[1,1])
ax7.plot(x2[:,1],v2[:,1],label='Orbit 1')
ax7.grid()
ax7.legend(loc='lower right')
ax7.set_xlabel(r'$y$')

ax8 = fig2.add_subplot(grid[0,2])
ax8.plot(x3[:,0],v3[:,0],label='Orbit 2')
ax8.grid()
ax8.legend(loc='lower right')
ax8.set_xlabel(r'$x$')

ax9 = fig2.add_subplot(grid[1,2])
ax9.plot(x3[:,1],v3[:,1],label='Orbit 2')
ax9.grid()
ax9.legend(loc='lower right')
ax9.set_xlabel(r'$y$')

fig2.tight_layout()
fig1.savefig('NonAxisSymetricPhaseSpace.pdf')

#%% Plot Total Energy
#width,height=SP.setupPlot(singleColumn=True)
#fig2 = plt.figure(figsize=(width,.5*height))
#grid2 = plt.GridSpec(1,2)
#
#ax4 = fig2.add_subplot(grid2[0,0])
#ax4.plot(t1,E1,label='Orbit 1')
#ax4.plot(t2,E2,label='Orbit 2')
#ax4.plot(t3,E3,label='Orbit 3')
#ax4.legend(loc='upper right')
#ax4.set_ylabel(r'$E_T$')
#ax4.set_xlabel('Time')
#ax4.grid()
#
##ax5 = fig2.add_subplot(grid2[0,1])
##ax5.plot(t1,L1,label='Orbit 1')
##ax5.plot(t2,L2,label='Orbit 2')
##ax5.plot(t3,L3,label='Orbit 3')
##ax5.legend(loc='upper right')
##ax5.set_ylabel(r'$L$')
##ax5.set_xlabel('Time')
#
##fig2.tight_layout()
##fig2.savefig('EnergyMomentumPlot.pdf')

#%% Save Data to csv file
names=np.array(['x'    ,'y'   , 'v_x','v_y'])
orbit1=np.array([x10[0],x10[1],v10[0],v10[1]])
orbit2=np.array([x20[0],x20[1],v20[0],v20[1]])
orbit3=np.array([x30[0],x30[1],v30[0],v30[1]])

array=[orbit1, orbit2, orbit3]
df = pd.DataFrame(array,columns=names)
df.to_csv('NonAxisSymetricIV.csv',
          float_format='%1.2f',
          index_label='Orbit')

