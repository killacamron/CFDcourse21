"""
Explicit Finite Difference Method Code
Solves the 2D Diffusion Equation
Written by: Cameron Armstrong (2020)
Institution: Virginia Commonwealth University

"""

import numpy
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

nx = 100
ny = 100
nt = 17
nu = .05
xl = 2
yl = 2
dx = xl / (nx - 1)
dy = yl / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu
T0 = 2
Tw = 1

if dx != dy:
    print('make sure step-sizes are equal')    

x = numpy.linspace(0, xl, nx)
y = numpy.linspace(0, yl, ny)
X, Y = numpy.meshgrid(x, y)

u = numpy.ones((ny, nx))  # create a 1xn vector of 1's
un = numpy.ones((ny, nx))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = T0  

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
surf = ax1.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)


ax1.set_xlim(0, xl)
ax1.set_ylim(0, yl)
ax1.set_zlim(1, 2.5)

ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$');

fig2  = plt.figure(figsize=(6,6), dpi=80)
ax2 = fig2.gca()
disp = plt.imshow(u[:])
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$');

def diffuse(nt):
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = T0  
    
    for n in range(nt + 1): 
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = Tw
        u[-1, :] = Tw
        u[:, 0] = Tw
        u[:, -1] = Tw
    
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    surf = ax1.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)
    ax1.set_zlim(1, 2.5)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$');
    fig2  = plt.figure(figsize=(6,6), dpi=80)
    ax2 = fig2.gca()
    disp = plt.imshow(u[:])
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$');
