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
    index=np.array(np.where(sgn)[0])
    indexv=np.array(np.where(v[index,1]>0)[0])
    index=index[indexv]
    N=len(index)
    xinterp=np.zeros(N)
    vxinterp=np.zeros(N)
    for n in range(N):
        i=index[n]
        xSP=np.array([x[i,0],x[i+1,0]])
        vxSP=np.array([v[i,0],v[i+1,0]])
        ySP=np.array([x[i,1],x[i+1,1]])
        if debug:
            print('index:',i)
            print('ySP:',ySP)
            print('xSP:',xSP)
        funcx=interpolate.interp1d(ySP,xSP)
        funcvx=interpolate.interp1d(ySP,vxSP)
        xinterp[n]=funcx(0)
        vxinterp[n]=funcvx(0)
    if debug:
        plt.plot(xinterp,vxinterp,'o')
    return xinterp,vxinterp,index

#%% Common parameters for all orbits
h=1e-3
A=0
B=100 #s

#%% Box Orbit
x10=np.array([.5,0])
v10=np.array([1,.1])

t1,x1,v1,U1,K1,E1=doleapfrog(x10,v10,A,B,h,False)
xss1,vxss1,index1=surfaceSection(x1,v1,False)

#%% Tube Orbit
x20=np.array([0,.2])
v20=np.array([1,0])

t2,x2,v2,U2,K2,E2=doleapfrog(x20,v20,A,B,h,False)
xss2,vxss2,index2=surfaceSection(x2,v2,False)

#%% Box orbit 2
x30=np.array([0,.1])
v30=np.array([1,0])

t3,x3,v3,U3,K3,E3=doleapfrog(x30,v30,A,B,h,False)
xss3,vxss3,index3=surfaceSection(x3,v3,False)

#%% Plot Box Orbit
width,height=SP.setupPlot(singleColumn=True)
fig1 = plt.figure(figsize=(.9*width,.5*height))
grid = plt.GridSpec(1,2)

ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(x1[:,0],x1[:,1])
ax1.grid()
ax1.set_ylabel(r'$y$')
ax1.set_xlabel(r'$x$')
ax1.set_title('Box Orbit 1')

ax2= fig1.add_subplot(grid[0,1])
ax2.plot(xss1,vxss1,'o')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$v_x$')
ax2.grid()
ax2.set_title('Surface of Section')


fig1.tight_layout()
fig1.savefig('LogPotentialBoxOrbitPlot.pdf')

#%% Plot Box Orbit 2
width,height=SP.setupPlot(singleColumn=True)
fig3 = plt.figure(figsize=(.9*width,.5*height))
grid = plt.GridSpec(1,2)

ax5 = fig3.add_subplot(grid[0,0])
ax5.plot(x3[:,0],x3[:,1])
ax5.grid()
ax5.set_ylabel(r'$y$')
ax5.set_xlabel(r'$x$')
ax5.set_title('Box Orbit 2')

ax6= fig3.add_subplot(grid[0,1])
ax6.plot(xss3,vxss3,'o')
ax6.set_xlabel(r'$x$')
ax6.set_ylabel(r'$v_x$')
ax6.grid()
ax6.set_title('Surface of Section')

fig3.tight_layout()
fig3.savefig('LogPotentialBoxOrbit2Plot.pdf')

#%% Plot Tube Orbit
width,height=SP.setupPlot(singleColumn=True)
fig2 = plt.figure(figsize=(.9*width,.5*height))
grid = plt.GridSpec(1,2)

ax3 = fig2.add_subplot(grid[0,0])
ax3.plot(x2[:,0],x2[:,1])
ax3.grid()
ax3.set_ylabel(r'$y$')
ax3.set_xlabel(r'$x$')
ax3.set_title('Tube Orbit')

ax4= fig2.add_subplot(grid[0,1])
ax4.plot(xss2,vxss2,'o')
ax4.set_xlabel(r'$x$')
ax4.set_ylabel(r'$v_x$')
ax4.grid()
ax4.set_title('Surface of Section')

fig2.tight_layout()
fig2.savefig('LogPotentialTubeOrbitPlot.pdf')

#%% Save Data to csv file
names=np.array(['x'    ,'y'   , '$v_x$','$v_y$','Time'])
indexNames=['Box Orbit','Tube Orbit ','Box Orbit 2']
row1=np.array([x10[0],x10[1],v10[0],v10[1],B])
row2=np.array([x20[0],x20[1],v20[0],v20[1],B])
row3=np.array([x30[0],x30[1],v30[0],v30[1],B])

rows=[row1, row2,row3]

df = pd.DataFrame(rows,columns=names,index=indexNames)

with open('LogPotentialIV.tex','w') as tf:
    tf.write(df.to_latex(float_format='%2.2f',
                         index=True,
                         escape=False))