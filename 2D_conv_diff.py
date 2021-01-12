"""
Explicit Finite Difference Method Code
Solves the 2D Temperature Convection-Diffusion Equation
Assumes Tubular Plug-Flow-Reactor in Laminar Regime 
Heat Source-Sink Included Uses Laminar Nusselt Correlation for "h"
Written by: Cameron Armstrong (2020)
Institution: Virginia Commonwealth University

"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
import math
from array import array

D = 0.0015875 # tubing diameter in m
xl = 30/100 # tubing length in m (cm/100) & x range
yl = D # tubing diameter & y range
nx = 300 # x grid resolution
ny = 50 # y grid resolution
dx = xl/(nx-1) # x step
dy = yl/(ny-1) # y step
k= .12 # thermal conductvity W/(m*K)
p = 1750 # density (kg/m^3)
Cp = 1172 # constant pressure specifc heat (J/kg/K)
a = k/(p*Cp) # alpha - thermal diffusivity m2/s
sigma = .01 # time step factor
dt = sigma * dx * dy / a # time step 
Vr = math.pi*(D/2)**2*xl # tubing volume in m
Qmlm = 1 # volumetric flowrate in mL/min
Q = (Qmlm*10**-6)/60 # volumetric flowrate in m3/s
Ac = math.pi*(D/2)**2 # cross-sectional area in m2
lamx = a*dt/dx**2 # lumped coefficient
lamy = a*dt/dy**2 # lumped coefficient
Nu = 3.66 # nusselt number laminar flow in tube
h = Nu*k/D # convective heat transfer coefficient (W/m2/K)
T0 = 130+273.15 # stream inlet temperature in degK
Tw = 25+273.15 # wall temperature in degK
reltol = 1e-9 # tolerance for convergence  

x = np.linspace(0, xl, nx) # x grid
y = np.linspace(0, yl, ny) # y grid
X, Y = np.meshgrid(x, y) # mesh of nodes

uAvg = Q/Ac # average velocity m/s
uMax = 2*uAvg # max velocity m/s
u = np.zeros(ny)
u[:] = np.linspace(-(D/2),(D/2),ny) # array intialization
u[:] = uMax*(1-(u[:]/(D/2))**2) # fully-developed hagan-poiselle profile
u[0]=u[-1]=0 # no slip BC
u = np.array([u,]*nx) # velocity field
u = u.T # transpose/align field

def lets_get_tubular(): 
    Ttol = np.zeros((ny,nx)) # Tolerance check array initialization
    T = np.ones((ny, nx))*Tw  # create a 1xn vector of 1's
    Tn = np.ones((ny, nx))*Tw # temporary solution array
    termcond = (np.abs((np.linalg.norm(Ttol)-np.linalg.norm(Tn))))/np.linalg.norm(Tn)
    stepcount = 1 # step counter
    while termcond >= reltol: # convergence-criteria/termination-condition
        termcond = np.abs((np.linalg.norm(Ttol)-np.linalg.norm(Tn)))/np.linalg.norm(Tn)
        Tn = T.copy()
        # FDM vectorized solution using explicit euler and CDS
        T[1:-1, 1:-1] = (Tn[1:-1,1:-1] - (u[1:-1,1:-1]*(dt/(2*dx))*(Tn[1:-1,2:]-Tn[1:-1,:-2]))+\
                         lamx *(Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) +\
                        lamy* (Tn[2:,1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1])) \
                         - h*D*math.pi*(Tn[1:-1,1:-1]-Tw)*dt/p/Cp*xl/Vr
        T[0, :] = Tw # tubing wall temp dirichlet BC
        T[-1, :] = Tw # tubing wall temp dirichlet BC
        T[:, 0] = T0 # inlet flow temp dirichlet BC
        T[:, -1] = T[:,-2] # outlet flow temp neumann BC
        Ttol=T.copy() # update solution
        stepcount += 1 # update counter
    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    surf = ax.plot_surface(X, Y, T[:], rstride=1, cstride=1, cmap=cm.viridis,
#        linewidth=0, antialiased=True)
#    ax.set_xlabel('$x$')
#    ax.set_ylabel('$y$');
    T[:]=T[:]-273.15
    fig1 = plt.subplot(211)
#    ax = fig1.gca()
#    plt.imshow(T[:])
    cont = plt.contourf(X,Y,T[:],50)
    ax = plt.gca()
    ax.axis('scaled')
    ax.axes.get_yaxis().set_visible(False)
    plt.xlim(0,.05)
    plt.xlabel('Tubing Length (m)')
    cbar = plt.colorbar(cont)
    cbar.ax.set_ylabel('Temperature (degC)')
    
    centerline = ny/2
    wallline = ny-5
    centerline = int(centerline)
    wallline = int(wallline)
    centerT = T[centerline,:]
    wallT = T[wallline,:]
    fig2 = plt.subplot(212)
    plt.plot(x, centerT,label='center')
    plt.plot(x,wallT,label='wall')
    plt.legend()
    plt.ylabel('Temperature (degC)')
    plt.xlabel('Tubing Length (m)')
    
    plt.show()
    print('Stepcount = %s' %(stepcount))