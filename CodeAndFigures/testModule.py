# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import NumericIntegrations as NI
import scipy.constants as constants

def dvdt(t,x,v):
  #Pendulum equation of motion
  dvdt=-constants.g*np.sin(x)
  return dvdt

e=1e-2
Ecrit=2*constants.g #
#rotation,libration, near unstable equilibrium
t0=np.array([0,0,0])
x0=np.array([-1.3*np.pi,.5,np.pi-e])
v0=np.array([.4,0,0])*constants.g

T=1/np.sqrt(constants.g) #Period
#

N=100000
a=0
b=T*10000
IV=np.concatenate((x0.reshape(3,1),v0.reshape(3,1)),
                  axis=1)

tRot,xRot,vRot=NI.leapfrog(dvdt,a,b,N,IV[0])