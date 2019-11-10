#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:00:37 2019

@author: Sandra Bustamante

Orbits in an axisymmetric potential

"""

import numpy as np
import matplotlib.pyplot as plt
import NumericIntegrations as NI
import SetupPlots as SP
import scipy.optimize as Opt
import pandas as pd

#%% For potential U=-(1+r**2)**(-1/2)

#Equation of motion is
def dvdt(t,x,v):
  #equation of motion for Tomre potential
  dvdt=-x*(1+(x**2).sum())**(-3/2)
  #print(dvdt)
  return dvdt

def potential(x):
    if x.ndim==1:
        U=-(1+(x**2).sum())**(-1/2)
    else:
        U=-(1+(x**2).sum(axis=1))**(-1/2)
    return U

def totalEnergy(x,v):
    if x.ndim==1:
        #assuming x=[x1,y1,z1]
        #same for v
        #U is toomre potential
        K=(1/2)*(v**2).sum()
    else:
        #assuming x=[[x1,y1,z1],[x2,y2,z2],...]
        #same for v
        K=(1/2)*(v**2).sum(axis=1)
    U=potential(x)
    E=U+K
    return E

def angularMomentum(x,v):
    if x.ndim!=1:
        r=np.linalg.norm(x,axis=1)
        thetaDot=(x[:,0]*v[:,1]-x[:,1]*v[:,0])/r**2
        L=(r**2)*thetaDot
    else:
        r=np.linalg.norm(x)
        thetaDot=(x[0]*v[1]-x[1]*v[0])/r**2
        L=(r**2)*thetaDot
    return L

def func(r,E,L):
    f=E+(1+r**2)**(-1/2)-((L/r)**2)/2
    return f

#%% Common parameters for all orbits
h=1e-3
a=0
b=200 #s
#%% First Orbit

#Setup initial values
x10=np.array([.3,0])
v10=np.array([.3,.4])
IV=np.concatenate((x10.reshape(1,2),v10.reshape(1,2)),axis=0)

#calculate conserved quantities
E1=totalEnergy(x10,v10)
L1=angularMomentum(x10,v10)

#Plot to determine range of roots
r1=np.linspace(0.001,1.5,1000)
f1=func(r1,E1,L1)
plt.plot(r1,f1)
plt.ylim(-1,1)
plt.grid()

#determine inner and outer radius
r1a=0.001
r1b=.4
r1c=1.2
r1min=Opt.bisect(func,r1a,r1b,args=(E1,L1,))
r1max=Opt.bisect(func,r1b,r1c,args=(E1,L1,))

#Integrate the orbit
t1,x1,v1=NI.leapfrog2D(dvdt,a,b,h,IV)
EL1=totalEnergy(x1,v1)
LL1=angularMomentum(x1,v1)

EL1=totalEnergy(x1,v1)
LL1=angularMomentum(x1,v1)

#%% Second Orbit

#Setup initial values
x20=np.array([0,.5])
v20=np.array([.6,0])
IV=np.concatenate((x20.reshape(1,2),v20.reshape(1,2)),axis=0)

#calculate conserved quantities
E2=totalEnergy(x20,v20)
L2=angularMomentum(x20,v20)

#Plot to determine range of roots
r2=np.linspace(0.001,1.5,1000)
f2=func(r2,E2,L2)
plt.plot(r2,f2)
plt.ylim(-1,1)
plt.grid()

#determine inner and outer radius
r2a=0.1
r2b=.6
r2c=1.2
r2min=Opt.bisect(func,r2a,r2b,args=(E2,L2,))
r2max=Opt.bisect(func,r2b,r2c,args=(E2,L2,))

#Integrate the orbit
t2,x2,v2=NI.leapfrog2D(dvdt,a,b,h,IV)

EL2=totalEnergy(x2,v2)
LL2=angularMomentum(x2,v2)


#U2,K2,E2=totalEnergy(x2,v2)
#L2=np.cross(x2,v2) #angular momentum rxv
#r2=1/np.sqrt(2*(E2-U2)/L2**2)

#%% Third Orbit

#Setup initial values
x30=np.array([1,0])
v30=np.array([0,1])
IV=np.concatenate((x30.reshape(1,2),v30.reshape(1,2)),axis=0)

#calculate conserved quantities
E3=totalEnergy(x30,v30)
L3=angularMomentum(x30,v30)

#Plot to determine range of roots
r3=np.linspace(0.1,7.5,100)
f3=func(r3,E3,L3)
plt.plot(r3,f3)
plt.ylim(-1,1)
plt.grid()

#determine inner and outer radius
r3a=0.01
r3b=2.5
r3c=7.5
r3min=Opt.bisect(func,r3a,r3b,args=(E3,L3,))
r3max=Opt.bisect(func,r3b,r3c,args=(E3,L3,))

#Integrate the orbit
t3,x3,v3=NI.leapfrog2D(dvdt,a,b,h,IV)

EL3=totalEnergy(x3,v3)
LL3=angularMomentum(x3,v3)

#U3,K3,E3=totalEnergy(x3,v3)
#L3=np.cross(x3,v3) #angular momentum rxv
#r3=1/np.sqrt(2*(E3-U3)/L3**2)

#%% Plot
width,height=SP.setupPlot(singleColumn=True)
fig1 = plt.figure(figsize=(.9*width,.5*height))
grid = plt.GridSpec(1,3)

theta=np.linspace(0,2*np.pi,100)

ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(x1[:,0],x1[:,1],label='Orbit 1')
ax1.plot(r1max*np.cos(theta),r1max*np.sin(theta))
ax1.plot(r1min*np.cos(theta),r1min*np.sin(theta))
ax1.legend(loc='lower right')
ax1.set_xlabel('x-position')
ax1.set_ylabel('y-position')
ax1.set_aspect('equal')
ax1.grid()

ax2 = fig1.add_subplot(grid[0,1])
ax2.plot(x2[:,0],x2[:,1],label='Orbit 2')
ax2.plot(r2max*np.cos(theta),r2max*np.sin(theta))
ax2.plot(r2min*np.cos(theta),r2min*np.sin(theta))
ax2.legend(loc='lower right')
ax2.set_xlabel('x-position')
ax2.set_aspect('equal')
ax2.grid()

ax3 = fig1.add_subplot(grid[0,2])
ax3.plot(x3[:,0],x3[:,1],label='Orbit 3')
ax3.plot(r3max*np.cos(theta),r3max*np.sin(theta))
ax3.plot(r3min*np.cos(theta),r3min*np.sin(theta))
ax3.legend(loc='lower right')
ax3.set_xlabel('x-position')
ax3.set_aspect('equal')
ax3.grid()

fig1.tight_layout()
fig1.savefig('ToomrePotentialOrbits.pdf')

#%% Plot Phase Space
#width,height=SP.setupPlot(singleColumn=True)
#fig1 = plt.figure(figsize=(width,.5*height))
#grid = plt.GridSpec(1,3)
#
#ax1 = fig1.add_subplot(grid[0,0])
#ax1.plot(np.linalg.norm(x1,axis=1),np.linalg.norm(v1,axis=1),label='Orbit 1')
#ax1.legend(loc='upper right')
#ax1.set_xlabel('r')
#ax1.set_ylabel(r'$v_r$')
#
#ax2 = fig1.add_subplot(grid[0,1])
#ax2.plot(np.linalg.norm(x2,axis=1),np.linalg.norm(v2,axis=1),label='Orbit 2')
#ax2.legend(loc='upper right')
#ax2.set_xlabel('r')
#
#ax3 = fig1.add_subplot(grid[0,2])
#ax3.plot(np.linalg.norm(x3,axis=1),np.linalg.norm(v3,axis=1),label='Orbit 3')
#ax3.legend(loc='upper right')
#ax3.set_xlabel('r')
#
#fig1.tight_layout()

#%% Plot Total Energy
width,height=SP.setupPlot(singleColumn=False)
fig2 = plt.figure(figsize=(width,2*height))
grid2 = plt.GridSpec(2,1)

ax4 = fig2.add_subplot(grid2[0,0])
ax4.plot(t1,EL1,label='Orbit 1')
ax4.plot(t2,EL2,label='Orbit 2')
ax4.plot(t3,EL3,label='Orbit 3')
#ax4.legend(loc='upper right')
ax4.legend()
ax4.set_ylabel(r'$E_T$')
ax4.set_xlabel('Time')

ax5 = fig2.add_subplot(grid2[1,0])
ax5.plot(t1,LL1,label='Orbit 1')
ax5.plot(t2,LL2,label='Orbit 2')
ax5.plot(t3,LL3,label='Orbit 3')
ax5.legend()
#ax5.legend(loc='upper right')
ax5.set_ylabel(r'$L$')
ax5.set_xlabel('Time')

fig2.tight_layout()
fig2.savefig('EnergyMomentumPlot.pdf')

#%% Save values to csv file
names=np.array(['x','y', '$v_x$','$v_y$',
                '$r_{inner}$','$r_{outer}$','$E_T$','$L$'])
indexNames=['Orbit 1','Orbit 2','Orbit 3']

row1=np.array([x10[0],x10[1],v10[0],v10[1],r1min,r1max,E1,L1])
row2=np.array([x20[0],x20[1],v20[0],v20[1],r2min,r2max,E2,L2])
row3=np.array([x30[0],x30[1],v30[0],v30[1],r3min,r3max,E3,L3])

rows=[row1, row2, row3]

df = pd.DataFrame(rows,columns=names,index=indexNames)

with open('ToomreOrbitsData.tex','w') as tf:
    tf.write(df.to_latex(float_format='%2.2f',
                         index=True,
                         escape=False))

#""" continue reading for formatters and
#how to apply special names to the indexes
#https://stackoverflow.com/questions/15069814/formatting-latex-to-latex-output
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_latex.html
#
#Code in Latex:
#\begin{table}[]
#    \centering
#    \input{CodeAndFigures/ToomreOrbitsData.tex}
#    \caption{Caption}
#    \label{tab:my_label}
#\end{table}
#
#"""


