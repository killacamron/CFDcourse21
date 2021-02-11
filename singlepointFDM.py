# -*- coding: utf-8 -*-
"""
Finite Difference Method Code for Forward, Backward, and Central Difference
Schemes for single points and basic polynomials.
Generates plots and calculates slope showing order of each method.
"""

import numpy as np
from matplotlib import pyplot as plt
import time
import scipy
from scipy.stats import linregress

def FDS(f,x,dx):
    return (f(x+dx)-f(x))/dx

def BDS(f,x,h):
    return (f(x)-f(x-dx))/dx

def CDS(f,x,h):
    return (f(x+dx)-f(x-dx))/(2*dx)

def ourFunc(x):
    return 4*x**3 - 5*x**2 +6*x -1

x = 1
dx = .1
numiter = 4

dfdt_FDS = FDS(ourFunc,x,dx)
dfdt_BDS = BDS(ourFunc,x,dx)
dfdt_CDS = CDS(ourFunc,x,dx)

exact = 12*x**2 - 10*x + 6

# print('Forward Difference Scheme = %s' %(dfdt_FDS))
# print('Backward Difference Scheme = %s' %(dfdt_BDS))
# print('Central Difference Scheme = %s' %(dfdt_CDS))
# print('Exact Solution = %s' %(exact))

# error_FDS = np.empty(0)
# error_BDS = np.empty(0)
# error_CDS = np.empty(0)
# dxarray = np.empty(0)

error_FDS = np.zeros(numiter)
error_BDS = np.zeros(numiter)
error_CDS = np.zeros(numiter)
dxarray = np.zeros(numiter)

for i in range(numiter):
    dfdt_FDS = FDS(ourFunc,x,dx)
    dfdt_BDS = BDS(ourFunc,x,dx)
    dfdt_CDS = CDS(ourFunc,x,dx)
    # error_FDS = np.append(error_FDS,abs(dfdt_FDS-exact))
    # error_BDS = np.append(error_BDS,abs(dfdt_BDS-exact))
    # error_CDS = np.append(error_CDS,abs(dfdt_CDS-exact))
    error_FDS[i] = abs(dfdt_FDS-exact)
    error_BDS[i] = abs(dfdt_BDS-exact)
    error_CDS[i] = abs(dfdt_CDS-exact)
    dxarray[i] = dx
    dx = dx/10
    
plt.figure()
plt.loglog(dxarray,error_FDS,label='Forward')
plt.loglog(dxarray,error_BDS,label='Backward')
plt.loglog(dxarray,error_CDS,label='Central')
plt.legend()

logerror_FDS = np.log(error_FDS)
logerror_BDS = np.log(error_BDS)
logerror_CDS = np.log(error_CDS)
logdx = np.log(dxarray)

order_FDS = scipy.stats.linregress(logdx,logerror_FDS)
order_BDS = scipy.stats.linregress(logdx,logerror_BDS)
order_CDS = scipy.stats.linregress(logdx,logerror_CDS)

print(order_FDS,order_BDS,order_CDS)
