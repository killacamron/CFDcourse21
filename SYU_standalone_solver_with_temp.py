# =============================================================================
# Explicit Finite Difference Method Code
# Solves the 1D Reaction-Advection-Diffusion Equation Coupled with Temperature
# Heat Source-Sink Included Uses Laminar Nusselt Correlation for "h"
# Assumes Tubular Plug-Flow-Reactor in Laminar Regime 
# Written by: Cameron Armstrong (2021)
# Institution: Virginia Commonwealth University

# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import genfromtxt

np.seterr(divide='ignore')

F_a = 1130
F_f = 373
D1 = .005
Q1 = 8.3*10**-8
D = 2*0.0015875 #tubing diameter in m
Ac1 = math.pi*(D/2)**2
V1 = 10.0*10**-6 #reactor volume in m^3
xl1 = V1/Ac1 #tubing length m
dt1 = .00005
nx1 =500
dx1 = xl1/(nx1-1)
x1 = np.linspace(0,xl1,nx1)
u1 = Q1/Ac1
Ea1 = 80256 #J/mol
k01 = 6.75e8 #m3/mol/s
# Ea2 = 90250 #J/mol
# k02 = 10 #m3/mol/s
# Ea1 = 805 #J/mol
# k01 = .06 #m3/mol/s
# Ea2 = 5386 #J/mol
# k02 = 0.13 #m3/mol/s
R = 8.314
k= .2501 #W/(m*K)
p = 786 #(kg/m^3)
Cp = 2200 #(J/kg/K)
Nu = 3.66 #nusselt number laminar flow in tube
h = Nu*k/D
T0 = 20+273.15
Tw = 20+273.15 
a = k/(p*Cp) #alpha - thermal diffusivity
lam = a*dt1/dx1**2 #heat transfer coefficient
#dHr = 0
dHr = -553000 #heat of reaction, J/mol


A = np.ones(nx1)
An = np.ones(nx1)
B = np.ones(nx1)
Bn = np.ones(nx1)
C = np.zeros(nx1)
Cn = np.zeros(nx1)
# S = np.zeros(nx1)
# Sn = np.zeros(nx1)
T = np.ones(nx1)*T0
Tn = np.ones(nx1)*T0
tolcheck = np.ones(nx1)
k1 = k01*np.exp(-Ea1/R/T[1:-1]) 
k2 = 0.0 #k02*np.exp(-Ea2/R/T[1:-1])
reltol = 1e-8 # tolerance for convergence  

def check_yourself(tolcheck,Cn):
    #global termcond
    out = np.abs(((np.linalg.norm(tolcheck)-np.linalg.norm(Cn)))/np.linalg.norm(Cn))
    return out

def CDS(u1,dt1,dx1,D1,k1,main,s1,s2,stoic):
    out = main[1:-1] -(u1)*(dt1/(2*dx1))*(main[2:]-main[:-2])+D1*dt1/dx1**2*(main[2:]-2*main[1:-1]+main[:-2])+stoic*k1*s1[1:-1]*s2[1:-1]*dt1   
    return out

def UDS_2(u1,dt1,dx1,D1,k1,main,s1,s2,stoic1,stoic2,k2):
    out = main[1:-1] -(u1)*(dt1/(dx1))*(main[1:-1]-main[:-2])+D1*dt1/dx1**2*(main[2:]-2*main[1:-1]+main[:-2]) \
            +stoic1*k1*s1[1:-1]**1.6*s2[1:-1]**0.5*dt1 +stoic2*k2*s1[1:-1]*s2[1:-1]*dt1        
    return out

def UDS(u1,dt1,dx1,D1,k1,main,s1,s2,stoic1):
    out = main[1:-1] -(u1)*(dt1/(dx1))*(main[1:-1]-main[:-2])+D1*dt1/dx1**2*(main[2:]-2*main[1:-1]+main[:-2]) \
            +stoic1*k1*s1[1:-1]**1.6*s2[1:-1]**0.5*dt1      
    return out

termcond = check_yourself(tolcheck,Cn)
while termcond >= reltol: #time loop
        termcond = check_yourself(tolcheck,Cn)
        T[0]=T0 # impose dirichlet BC
        T[nx1-1]=T[nx1-2] # impose neumann BC
        Tn = T.copy() # update temporary array
        # next line is vectorized FDM spatial solution
        T[1:-1] = Tn[1:-1]-(u1*(dt1/dx1)*(Tn[1:-1]                     \
                    -Tn[:-2]))+lam*(Tn[2:]-2*Tn[1:-1]+Tn[:-2])      \
                    -h*D*math.pi*(Tn[1:-1]-Tw)*dt1/p/Cp*xl1/V1        \
                    -dHr*k1*A[1:-1]**1.6*B[1:-1]**0.5/p/Cp*dt1
        k1 = k01*np.exp(-Ea1/R/T[1:-1])
        #k2 = k02*np.exp(-Ea2/R/T[1:-1])
        An = A.copy()
        Bn = B.copy()
        Cn = C.copy()
        # Sn = S.copy()
        A[0] = F_a
        B[0] = F_f
        A[nx1-1] = A[nx1-2] 
        B[nx1-1] = B[nx1-2] 
        C[nx1-1] = C[nx1-2] 
        # S[nx1-1] = S[nx1-2]
        A[1:-1] = UDS(u1,dt1,dx1,D1,k1,An,An,Bn,-1)
        B[1:-1] = UDS(u1,dt1,dx1,D1,k1,Bn,An,Bn,-1)
        C[1:-1] = UDS(u1,dt1,dx1,D1,k1,Cn,An,Bn,1)
        #S[1:-1] = UDS(u1,dt1,dx1,D1,k2,Sn,An,Bn,0,1,k2)
        tolcheck = C.copy()


plt.rcParams["font.family"] = "sans-serif"
plt.rc('axes',labelsize=18)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('figure',titlesize=10)
plt.rc('legend',fontsize=16)
              
plt.figure(1)
plt.plot(x1/xl1,C,label='Product')
plt.plot(x1/xl1,A,label='Reagent1')
plt.plot(x1/xl1,B,label='Reagent2')
#plt.plot(x1/xl1,S,label='Side Product')
plt.legend(loc=7)
plt.xlabel('Reactor Length (m)')
plt.ylabel('Molar Concentration (mol/m3)') 

plt.figure(2)
plt.plot(x1/xl1,T-273.15)
plt.axvline(x=.05/xl1,color='k',linestyle='--')
plt.xlabel('Reactor Length (m)')
plt.ylabel('Reactor Temperature (degC)')

# T3 = T[:]
# plt.figure(3)
# plt.plot(x1/xl1,T1-273.15,label='25 degC Wall Temp')
# plt.plot(x1/xl1,T2-273.15,label='-10 degC Wall Temp')
# plt.plot(x1/xl1,T3-273.15,label='-10 degC Wall Temp + precooling')
# plt.axvline(x=.05/xl1,color='k',linestyle='--')
# plt.xlabel('Reactor Length (m)')
# plt.ylabel('Reactor Temperature (degC)')
# plt.legend(loc=1)

print('yield sulfonamide = %s'%(C[nx1-1]/F_f))