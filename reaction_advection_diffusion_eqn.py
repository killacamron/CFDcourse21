"""
Explicit Finite Difference Method Code
Solves the 1D Isothermal Reaction-Advection-Diffusion Equation
Assumes Tubular Plug-Flow-Reactor in Laminar Regime 
Written by: Cameron Armstrong (2020)
Institution: Virginia Commonwealth University

"""
import numpy as np
from matplotlib import pyplot as plt

np.seterr(divide='ignore')

F_a = 600
F_f = 500
D1 = .005
Q1 = 8.3*10**-8
Ac1 = 2.0*10**-6
V1 = 1.0*10**-5 #reactor volume in m^3
xl1 = V1/Ac1 #tubing length m
dt1 = .0075
nx1 = 200
dx1 = xl1/(nx1-1)
x1 = np.linspace(0,xl1,nx1)
u1 = Q1/Ac1
Ea1 = 53106.5 #J/mol
k01 = 1175.26 #m3/mol/s
R = 8.314
T1 = 150 + 273.15


A = np.ones(nx1)
An = np.ones(nx1)
B = np.ones(nx1)
Bn = np.ones(nx1)
C = np.zeros(nx1)
Cn = np.zeros(nx1)
tolcheck = np.ones(nx1)
k1 = k01*np.exp(-Ea1/R/T1)
reltol = 1e-9 # tolerance for convergence  

def check_yourself(tolcheck,Cn):
    #global termcond
    out = (np.abs((np.linalg.norm(tolcheck)-np.linalg.norm(Cn))))/np.linalg.norm(Cn) 
    return out

def CDS(u1,dt1,dx1,D1,k1,main,s1,s2,stoic):
    out = main[1:-1] -(u1)*(dt1/(2*dx1))*(main[2:]-main[:-2])+D1*dt1/dx1**2*(main[2:]-2*main[1:-1]+main[:-2])+stoic*k1*s1[1:-1]*s2[1:-1]*dt1   
    return out

def UDS(u1,dt1,dx1,D1,k1,main,s1,s2,stoic):
    out = main[1:-1] -(u1)*(dt1/(dx1))*(main[1:-1]-main[:-2])+D1*dt1/dx1**2*(main[2:]-2*main[1:-1]+main[:-2])+stoic*k1*s1[1:-1]*s2[1:-1]*dt1
    return out

termcond = check_yourself(tolcheck,Cn)
while termcond >= reltol: #time loop
        termcond = check_yourself(tolcheck,Cn)
        An = A.copy()
        Bn = B.copy()
        Cn = C.copy()
        A[0] = F_a
        B[0] = F_f
        A[nx1-1] = A[nx1-2] 
        B[nx1-1] = B[nx1-2] 
        C[nx1-1] = C[nx1-2] 
        A[1:-1] = UDS(u1,dt1,dx1,D1,k1,An,An,Bn,-1)
        B[1:-1] = UDS(u1,dt1,dx1,D1,k1,Bn,An,Bn,-1)
        C[1:-1] = UDS(u1,dt1,dx1,D1,k1,Cn,An,Bn,1)
        tolcheck = C.copy()
              
plt.figure(1)
plt.plot(x1/xl1,C)
plt.plot(x1/xl1,A)
plt.plot(x1/xl1,B)
plt.xlabel('Normalized Reactor Length (-)')
plt.ylabel('Molar Concentration R1 (mol/m3)') 