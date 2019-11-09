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
import scipy.interpolate as interpolate

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

def dvdt(t,X,V):
    v0=1
    Rc=0.14
    q=0.9
    x=X[0]
    y=X[1]
    den=Rc**2 + x**2 + (y/q)**2
    dvxdt=((-v0**2)*x)/den
    dvydt=((-v0**2)*y)/(den*q**2)
    dvdt=np.array([dvxdt,dvydt])
    return dvdt

def doleapfrog(x0,v0,A,B,h,debug=False):
    IV=np.concatenate((x0.reshape(1,2),v0.reshape(1,2)),axis=0)
    t,x,v=NI.leapfrog2D(dvdt,A,B,h,IV)
    U,K,E=totalEnergy(x,v)
    if debug:
        print('E:',E[1],E[-1])
        print('U:',U[1],U[-1])
        print('K:',K[1],K[-1])
        plt.plot(x[:,0],x[:,1])

    return t,x,v,U,K,E

def surfaceSection(x,v,debug=False):
    m=x[1:,1]
    s=x[:-1,1]
    sgn=(m*s)<0
    index=np.argwhere(sgn)
    N=len(index)
    xinterp=np.zeros(N)
    vxinterp=np.zeros(N)
    for n in range(N):
        i=index[n]
        xSP=np.concatenate((x[i,0],x[i+1,0]))
        vxSP=np.concatenate((v[i,0],v[i+1,0]))
        ySP=np.concatenate((x[i,1],x[i+1,1]))
        funcx=interpolate.interp1d(ySP,xSP)
        funcvx=interpolate.interp1d(ySP,vxSP)
        xinterp[n]=funcx(0)
        vxinterp[n]=funcvx(0)
    if debug:
        plt.plot(xinterp,vxinterp,'o')
    return xinterp,vxinterp

#%% Common parameters for all orbits
h=1e-3
A=0
B=100 #s

#%% Box Orbit
x10=np.array([.5,0])
v10=np.array([1,.1])
#x10=np.array([0,.5])
#v10=np.array([.2,0])

t1,x1,v1,U1,K1,E1=doleapfrog(x10,v10,A,B,h,True)
xss1,vxss1=surfaceSection(x1,v1,False)

#%% Tube Orbit
x20=np.array([0,.2])
v20=np.array([1,0])

t2,x2,v2,U2,K2,E2=doleapfrog(x20,v20,A,B,h,True)
xss2,vxss2=surfaceSection(x2,v2,False)

#%% Plot Box Orbit
width,height=SP.setupPlot(singleColumn=True)
fig1 = plt.figure(figsize=(.9*width,.5*height))
grid = plt.GridSpec(1,2)

ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(x1[:,0],x1[:,1])
ax1.grid()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$x$')

ax2= fig1.add_subplot(grid[0,1])
ax2.plot(xss1,vxss1,'o')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$v_x$')
ax2.grid()

fig1.tight_layout()
fig1.savefig('LogPotentialBoxOrbitPlot.pdf')

#%% Plot Tube Orbit
width,height=SP.setupPlot(singleColumn=True)
fig2 = plt.figure(figsize=(.9*width,.5*height))
grid = plt.GridSpec(1,2)

ax3 = fig2.add_subplot(grid[0,0])
ax3.plot(x2[:,0],x2[:,1])
ax3.grid()
ax3.set_ylabel(r'$y$')
ax3.set_xlabel(r'$x$')

ax4= fig2.add_subplot(grid[0,1])
ax4.plot(xss2,vxss2,'o')
ax4.set_xlabel(r'$x$')
ax4.set_ylabel(r'$v_x$')
ax4.grid()

fig2.tight_layout()
fig2.savefig('LogPotentialTubeOrbitPlot.pdf')

#%% Save Data to csv file
names=np.array(['x'    ,'y'   , '$v_x$','$v_y$','Time'])
indexNames=['Box Orbit','Tube Orbit ']
row1=np.array([x10[0],x10[1],v10[0],v10[1],B])
row2=np.array([x20[0],x20[1],v20[0],v20[1],B])

rows=[row1, row2]

df = pd.DataFrame(rows,columns=names,index=indexNames)

with open('LogPotentialIV.tex','w') as tf:
    tf.write(df.to_latex(float_format='%2.2f',
                         index=True,
                         escape=False))