1# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:47:09 2021

@author: Cam
"""

# =============================================================================
# 
# Explicit Finite Difference Method Code
# Solves the 2D Temperature Convection-Diffusion Equation
# Assumes Tubular Plug-Flow-Reactor in Laminar Regime 
# Assumes hagen poiseuille velocity profile
# Heat Source-Sink Included Uses Laminar Nusselt Correlation for "h"
# Written by: Cameron Armstrong (2020)
# Institution: Virginia Commonwealth University
# 
# =============================================================================

# Required Modules
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
import math
from datetime import datetime as date

D = 0.0015875                                   # tubing diameter in m
zl = 30/100                                     # tubing length in m & x range
rl = D/2                                          # tubing diameter & y range
nz = 300                                        # x grid points
nr = 50                                         # y grid points
dz = zl/(nz-1)                                  # x stepsize
dr = rl/(nr-1)                                  # y stepsize
k= .12                                          # thermal conductvity W/(m*K)
p = 1750                                        # density (kg/m3)
Cp = 1172                                       # specifc heat (J/kg/K)
a = k/(p*Cp)                                    # thermal diffusivity (m2/s)
sigma = .001                                     # time step factor
dt = sigma*dz*dr/a                              # time stepsize 
dt = .0005
Vr = math.pi*(D/2)**2*zl                        # tubing volume (m3)
Qmlm = 4                                       # volumetric flowrate (mL/min)
Q = (Qmlm*10**-6)/60                            # volumetric flowrate (m3/s)
Ac = math.pi*(D/2)**2                           # cross-sectional area (m2)
lamz = a*dt/dz**2                               # lumped coefficient
lamr = a*dt/dr**2                               # lumped coefficient
D1 = 10e-9 # axial dispersion coefficient (m^2/s)
clamz = D1*dt/dz**2                               # lumped coefficient
clamr = D1*dt/dr**2                               # lumped coefficient
Nu = 3.66                                       # nusselt laminar flow in tube
h = Nu*k/D                                      # convective heat transfer coefficient (W/m2/K)
T0 = 25+273.15                                 # stream inlet temperature (degK)
Tw = 130+273.15                                  # wall temperature (degK)
reltol = 1e-8                                   # tolerance for convergence  
Rgc = 8.314

Ea1 = 53106.5 # activation energy (J/mol)
k01 = 1175.26 # arrhenius factor (m3/mol/s)
A0 = 600 # stock concentration of species A (acrylate) (mol/m3)
B0 = 500 # stock concentration of species B (fluoro) (mol/m3)

# grid formation
z = np.linspace(0, zl, nz) 
r = np.linspace(0, rl, nr) 
Z, R = np.meshgrid(z, r) 

# hagen poiseuille velocity field generation
uAvg = Q/Ac                                     # average velocity (m/s)
uMax = 2*uAvg                                   # max velocity (m/s)
u = np.zeros(nr)                                # array initilization
u[:] = np.linspace(0,rl,nr)             # array intialization
u[:] = uMax*(1-(u[:]/(D/2))**2)                 # hagan-poiselle profile
u[-1] = 0                                 # no slip BC
u = np.array([u,]*nz)                           # velocity field
u = u.T                                         # transpose/align field
maxCFL = np.max(u*dt/dz)                        # CFL condition calc.

T = np.ones((nr, nz))  
Tn = np.ones((nr, nz)) 
A = np.ones((nr, nz))  
An = np.ones((nr, nz)) 
B = np.ones((nr, nz))  
Bn = np.ones((nr, nz)) 
C = np.ones((nr, nz))  
Cn = np.ones((nr, nz)) 
tolcheck = np.zeros((nr,nz))
print('The max CFL is %s'%(maxCFL))

def check_yourself(tolcheck,var):
    #global termcond
    out = np.abs(((np.linalg.norm(tolcheck)-np.linalg.norm(var)))/np.linalg.norm(var))
    return out
  
def impose_2D_BC_T(var,val1,val2):
    var[-1, :] = val1 # tubing wall temp dirichlet BC
    var[0, :] = var[1,:] #symmetry condition
    var[:, 0] = val2 # inlet flow temp dirichlet BC
    var[:, -1] = var[:,-2] # outlet flow temp neumann BC
    return var
  
def impose_2D_BC_Conc(var,val1,val2):
  var[-1,:] = var[-2,:] # tubing wall temp dirichlet BC
  var[0, :] = var[1,:] #symmetry condition
  var[:, 0] = val2 # inlet flow temp dirichlet BC
  var[:, -1] = var[:,-2] # outlet flow temp neumann BC
  return var

# main function loop
def lets_get_tubular(): 
    init = date.now()
    # array initialization
    T = np.ones((nr, nz))  
    Tn = np.ones((nr, nz)) 
    A = np.ones((nr, nz))  
    An = np.ones((nr, nz)) 
    B = np.ones((nr, nz))  
    Bn = np.ones((nr, nz)) 
    C = np.ones((nr, nz))  
    Cn = np.ones((nr, nz)) 
    tolcheck = np.zeros((nr,nz))
    # initialize termination condition
    # compares norms of current and previous solution arrays
    termcond = check_yourself(tolcheck,Cn)
    stepcount = 1 # step counter
    while termcond >= reltol: 
        termcond = check_yourself(tolcheck,Cn)
        T[:] = impose_2D_BC_T(T,Tw,T0)
        Tn = T.copy()
        # FDM vectorized solution using explicit euler and CDS
        T[1:-1, 1:-1] = (Tn[1:-1,1:-1] - (u[1:-1,1:-1]*(dt/(2*dz))*(Tn[1:-1,2:]  \
                     -Tn[1:-1,:-2])) + a*dt/(2*dr)/R[1:-1,1:-1]*(Tn[2:,1:-1]-Tn[:-2,1:-1])     \
                     + lamz *(Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) \
                     + lamr* (Tn[2:,1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1])) \
                     - h*D*math.pi*(Tn[1:-1,1:-1]-Tw)*dt/p/Cp*zl/Vr  
                     
        k1 = k01*np.exp(-Ea1/Rgc/T[:])
        An = A.copy()
        Bn = B.copy()
        Cn = C.copy()
        A[:] = impose_2D_BC_Conc(A,0.0,A0)
        B[:] = impose_2D_BC_Conc(B,0.0,B0)
        C[:] = impose_2D_BC_Conc(C,0.0,0.0)
        
        A[1:-1, 1:-1] = (An[1:-1,1:-1] - (u[1:-1,1:-1]*(dt/(2*dz))*(An[1:-1,2:]  \
                     -An[1:-1,:-2])) + D1*dt/(2*dr)/R[1:-1,1:-1]*(An[2:,1:-1]-An[:-2,1:-1])     \
                     + clamz *(An[1:-1, 2:] - 2 * An[1:-1, 1:-1] + An[1:-1, :-2]) \
                     + clamr* (An[2:,1:-1] - 2 * An[1:-1, 1:-1] + An[:-2, 1:-1])) \
                     - k1[1:-1,1:-1]*(An[1:-1,1:-1]*Bn[1:-1,1:-1])*dt
        
        B[1:-1, 1:-1] = (Bn[1:-1,1:-1] - (u[1:-1,1:-1]*(dt/(2*dz))*(Bn[1:-1,2:]  \
                     -Bn[1:-1,:-2])) + D1*dt/(2*dr)/R[1:-1,1:-1]*(Bn[2:,1:-1]-Bn[:-2,1:-1])     \
                     + clamz *(Bn[1:-1, 2:] - 2 * Bn[1:-1, 1:-1] + Bn[1:-1, :-2]) \
                     + clamr* (Bn[2:,1:-1] - 2 * Bn[1:-1, 1:-1] + Bn[:-2, 1:-1])) \
                     -  k1[1:-1,1:-1]*(An[1:-1,1:-1]*Bn[1:-1,1:-1])*dt
                     
        C[1:-1, 1:-1] = (Cn[1:-1,1:-1] - (u[1:-1,1:-1]*(dt/(2*dz))*(Cn[1:-1,2:]  \
                     -Cn[1:-1,:-2])) + D1*dt/(2*dr)/R[1:-1,1:-1]*(Cn[2:,1:-1]-Cn[:-2,1:-1])     \
                     + clamz *(Cn[1:-1, 2:] - 2 * Cn[1:-1, 1:-1] + Cn[1:-1, :-2]) \
                     + clamr* (Cn[2:,1:-1] - 2 * Cn[1:-1, 1:-1] + Cn[:-2, 1:-1])) \
                     +  k1[1:-1,1:-1]*(An[1:-1,1:-1]*Bn[1:-1,1:-1])*dt              
        tolcheck = C.copy()
        #wallflux = k*(T[-2, :] - T[-1,:])/dr/1000
        stepcount += 1 # update counter

    end = date.now()
    cputime = end-init
    cputime = cputime.total_seconds()
    print(cputime)
    T[:]=T[:]-273.15                            # converts back to degC
    
    # generates plots
    # top plot is 2D filled contour plot 
    # bottom plot is centerline and near-wall line data points
    fig1 = plt.subplot(311)
#    ax = fig1.gca()
#    plt.imshow(T[:])
    cont = plt.contourf(Z,R,T[:],50)
    ax = plt.gca()
    #ax.axis('scaled')
    ax.axes.get_yaxis().set_visible(True)
    plt.xlim(0,zl)
    plt.yticks([0,D/2],['Center (r = 0)','Wall (r = R)'],fontsize ='10')
    plt.xlabel('Tubing Length (m)')
    cbar = plt.colorbar(cont)
    cbar.ax.set_ylabel('Temperature (degC)')
    
    centerline = nr/2
    wallline = nr-5
    centerline = int(centerline)
    wallline = int(wallline)
    centerT = T[centerline,:]
    wallT = T[wallline,:]
    
    fig2 = plt.subplot(312)
    plt.plot(z, centerT,label='center')
    plt.plot(z,wallT,label='wall')
    plt.legend()
    plt.ylabel('Temperature (degC)')
    plt.xlabel('Tubing Length (m)')
    
    fig3 = plt.subplot(313)
#    ax = fig1.gca()
#    plt.imshow(T[:])
    cont = plt.contourf(Z,R,C[:],50)
    ax = plt.gca()
    #ax.axis('scaled')
    ax.axes.get_yaxis().set_visible(True)
    plt.xlim(0,zl)
    plt.yticks([0,D/2],['Center (r = 0)','Wall (r = R)'],fontsize ='10')
    plt.xlabel('Tubing Length (m)')
    cbar = plt.colorbar(cont)
    cbar.ax.set_ylabel('Concentration (mol/m3)')
    
    plt.show()
    print('Stepcount = %s' %(stepcount))
#    
#    plt.figure(2)
#    plt.plot(z,wallflux)
    
    plt.figure(2)
    plt.plot(u[:,nz-1])
    
#    
if __name__ == "__main__":
    lets_get_tubular()