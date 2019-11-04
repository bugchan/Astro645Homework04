#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:08:16 2019

@author: sbustamanteg
"""

import numpy as np

def leapfrog(dvdt,a,b,h,IV):
  t = np.arange(a,b,h)
  N=int((b-a)/h)
  x=np.zeros(N,)
  v=np.zeros(N)
  x[0], v[0] = IV
  for n in np.arange(1,N):
    k1=x[n-1]+0.5*h*v[n-1]
    v[n]=v[n-1]+h*dvdt(t,k1,v)
    x[n]=k1+0.5*h*v[n]
  return t, x, v

def RK4(dvdt, a, b, h, IV):
    t = np.arange(a,b,h)  # create mesh
    N=int((b-a)/h)
    x = np.zeros(N)       # initialize x
    v = np.zeros(N)       # initialize x
    x[0], v[0] = IV       # set initial values
    # apply Fourth Order Runge-Kutta Method
    for i in np.arange(1,N):
        k1= h*dvdt(t[i-1], x[i-1], v[i-1])
        j1= h*v[i-1]
        k2= h*dvdt(t[i-1]+h/2.0, x[i-1]+j1/2.0,
                   v[i-1]+k1/2.0)
        j2= h*(v[i-1]+k1/2.0)
        k3= h*dvdt(t[i-1]+h/2.0, x[i-1]+j2/2.0,
                   v[i-1]+k2/2.0)
        j3= h*(v[i-1]+k2/2.0)
        k4= h*dvdt( t[i], x[i-1] + j3, v[i-1] + k3)
        j4= h*(v[i-1] + k3)
        v[i] =v[i-1] + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        x[i] =x[i-1] + (j1 + 2.0*j2 + 2.0*j3 + j4)/6.0
    return t, x, v
