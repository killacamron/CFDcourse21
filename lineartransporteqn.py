import numpy as np
from matplotlib import pyplot as plt

"""
Explicit Finite Difference Method Code to Solve the 1D Linear Transport Equation
Written by: Cameron Armstrong (2019)
Institution: Virginia Commonwealth University

"""
xl = 2 # x length
nx = 500 # number of grid points
x = np.linspace(0,xl,nx) # x grid 
dx = xl/(nx-1) # x stepsize
nt = 150 # number of timesteps
dt = 0.0025 # time stepsize
c = 1 # wave speed
g = .01 # gaussian variance parameter (peak width)
theta = x/(0.5*xl) # gaussian mean parameter (peak position)

u = np.ones(nx) # initializing solution array
un = np.ones(nx) # initializing temporary solution array
u = (1/(2*np.sqrt(np.pi*(g))))*np.exp(-(1-theta)**2/(4*g)) # initial condition (IC) as a gaussian
plt.plot(x,u); # plots IC

#BDS/Upwind with inner for-loop
#for n in range(nt):
#    un = u.copy()
#    for i in range(1,nx-1):
#        u[i] = un[i] - c*dt/(dx)*(un[i]-un[i-1])

#BDS/Upwind with vectorization
for n in range(nt):
    un = u.copy()
    u[1:-1] = un[1:-1] - c*dt/(dx)*(un[1:-1]-un[:-2])

#CDS with inner for-loop
#for n in range(nt):
#    un = u.copy()
#    for i in range(1,nx-1):
#        u[i] = un[i] - c*dt/(2*dx)*(un[i+1]-un[i-1])

#CDS with vectorization
#for n in range(nt):
#    un = u.copy()
#    u[1:-1] = un[1:-1] - c*dt/(2*dx)*(un[2:]-un[:-2])

plt.plot(x,u);        