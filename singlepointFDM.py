# =============================================================================
# 
# Finite Difference Method Code for Forward, Backward, and Central Difference
# Schemes for single points and user-defined functions
# User must provide exact solution for derivative
# Generates plots and calculates slope showing order of each method
# Written by: Cameron Armstrong (2021)
# Institution: Virginia Commonwealth University
# =============================================================================

# Required Modules
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.stats import linregress

# Difference Schemes Functions
# f is function input
# x is point input
# h is stepsize

def FDS(f,x,h):
    return (f(x+h)-f(x))/h

def BDS(f,x,h):
    return (f(x)-f(x-h))/h

def CDS(f,x,h):
    return (f(x+h)-f(x-h))/(2*h)

def ourFunc(x):
    return 4*x**3 - 5*x**2 +6*x -1

# Variable Definitions
x = 1                                           # point to analyze
dx = .01                                        # initial step-size
numiter = 4                                     # number of iterations
factor = 10                                     # stepsize factor - default is 10

# function calls
dfdt_FDS = round(FDS(ourFunc,x,dx),1)          
dfdt_BDS = round(BDS(ourFunc,x,dx),1)
dfdt_CDS = round(CDS(ourFunc,x,dx),1)

exact = 12*x**2 - 10*x + 6                      # exact solution

# prints single point derivative approximations at initial stepsize
# comment these out if you just want to run the loop below
print('Forward Difference Scheme = %s' %(dfdt_FDS))
print('Backward Difference Scheme = %s' %(dfdt_BDS))
print('Central Difference Scheme = %s' %(dfdt_CDS))
print('Exact Solution = %s' %(exact))

# error arrays and stepsize array initialization
error_FDS = np.zeros(numiter)
error_BDS = np.zeros(numiter)
error_CDS = np.zeros(numiter)
dxarray = np.zeros(numiter)

# loop for going through incrementally smaller step sizes
for i in range(numiter):
    # function calls
    dfdt_FDS = FDS(ourFunc,x,dx)
    dfdt_BDS = BDS(ourFunc,x,dx)
    dfdt_CDS = CDS(ourFunc,x,dx)
    # calculating error
    error_FDS[i] = abs(dfdt_FDS-exact)
    error_BDS[i] = abs(dfdt_BDS-exact)
    error_CDS[i] = abs(dfdt_CDS-exact)
    # storing current stepsize
    dxarray[i] = dx
    # update stepsize
    dx = dx/factor
    
# plotting of log(error) vs log(stepsize)
plt.figure()
plt.loglog(dxarray,error_FDS,label='Forward')
plt.loglog(dxarray,error_BDS,label='Backward')
plt.loglog(dxarray,error_CDS,label='Central')
plt.xlabel('log(stepsize)')
plt.ylabel('log(error)')
plt.legend()

# calculating log of errors and stepsize
logerror_FDS = np.log(error_FDS)
logerror_BDS = np.log(error_BDS)
logerror_CDS = np.log(error_CDS)
logdx = np.log(dxarray)

# linear regression on log-log data
# returns approxmate order of each method rounded to neareast integer
order_FDS = scipy.stats.linregress(logdx,logerror_FDS)
order_FDS = round(order_FDS.slope,0)
order_BDS = scipy.stats.linregress(logdx,logerror_BDS)
order_BDS = round(order_BDS.slope,0)
order_CDS = scipy.stats.linregress(logdx,logerror_CDS)
order_CDS = round(order_CDS.slope,0)

# prints apprimxate scheme orders
print('Order of FDS = %s' %(order_FDS))
print('Order of BDS = %s' %(order_BDS))
print('Order of CDS = %s' %(order_CDS))

