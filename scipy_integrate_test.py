# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:12:58 2021

@author: Cam
"""

import scipy
from scipy import integrate
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np

#x2 = lambda x: x**2
#integral = integrate.quad(x2,0,4)
#exact = 4**3/3.0
#print(integral)
#print(exact)

# def expon_decay(t,y):
#     return -0.5*y

# sol = integrate.solve_ivp(expon_decay,[0,10],[10])

# y = sol.y.T
# t = sol.t
# plt.plot(t,y)

# def lotkavolterra(t, z, a, b, c, d):
#     x, y = z
#     return [a*x - b*x*y, -c*y + d*x*y]

# sol = solve_ivp(lotkavolterra, [0, 15], [10, 2], args=(1.5, 1, 3, 1),
#                 dense_output=True)
# t = np.linspace(0, 15, 300)
# z = sol.sol(t)
# plt.plot(t, z.T)
# plt.xlabel('t')
# plt.legend(['x', 'y'], shadow=True)
# plt.title('Lotka-Volterra System')
# plt.show()

h = 0.01
t = np.array([0.0, 10.0])
yinit = np.array([0.4, -0.7, 21.0])

def myFunc(t, y):
    # Lorenz system
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0

    dy = np.zeros((len(y)))

    dy[0] = sigma*(y[1] - y[0])
    dy[1] = y[0]*(rho - y[2]) - y[1]
    dy[2] = y[0]*y[1] - beta*y[2]

    return dy

sol_lorenz = solve_ivp(myFunc,t,yinit,max_step=.01,method='RK45')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_lorenz.y[0], sol_lorenz.y[1], sol_lorenz.y[2])
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)