#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:24:55 2019

@author: sbustamanteg
"""
import numpy as np
import matplotlib.pyplot as plt

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
            'axes.axisbelow' : True,
            'pgf.rcfonts' : False,
            }
  plt.rcParams.update(params)
  return width,height

#def 3x1Plot():
