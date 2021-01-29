"""
Newton's Method Code with Armijo Line Rule
Written by: Cameron Armstrong (2020)
Institution: Virginia Commonwealth University

"""
import numpy as np
from matplotlib import pyplot as plt

#Newtons Method with armijo rule

nx = 500
x = np.linspace(-10,10,nx+1)
y = np.ones(nx+1)
fplot = np.ones(nx+1)
fplot = fplot.astype(float)
fplot[:] = np.cos(x[:])-x[:]
xguess = 10
xnewroot = xguess
xoldroot = xguess
tol = 1e-9

def f_func(f_in):
    global f_func,f
    f = np.cos(f_in)-f_in
    #f = np.arctan(f_in)
    return f

def fprime_func(fp_in):
    global fprime_func,fprime
    fprime = -np.sin(fp_in)-1
    #fprime = 1/(1+fp_in**2)
    return fprime

f_func(xguess)
fprime_func(xguess)
d = -f/fprime
s = 1
s_armijo = 0
j = 1
r0 = np.linalg.norm(f)    
r = np.linalg.norm(f)
endcond = tol*r0+tol
j = 0
while r >= endcond:
    f = f_func(xoldroot)
    fprime = fprime_func(xoldroot)
    d = -f/fprime
    y[:] = fprime*(x[:]-xoldroot)+f
    xnewroot = xoldroot + s*d
    normf = np.linalg.norm(f_func(xoldroot))
    normft = np.linalg.norm(f_func(xnewroot))
    r = np.linalg.norm(f)
    j = j+1
    if normft < normf:
       xoldroot = xnewroot
       s = 1
    else:
        s = s/2
    plt.figure(1)
    plt.semilogy(j,r,marker='*',color='black')
    plt.xlabel('iteration')
    plt.ylabel('relative residual')
   

func_root = np.round(xnewroot,3)
print('The converged funtion root is %s after %s iterations' %(func_root,j))
    
plt.subplot(121)
plt.plot(x,fplot)
plt.plot(x,y)
plt.scatter(func_root,0,marker='o',color='blue')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlim(-5,5)
plt.ylim(-10,10)   
plt.subplot(122)
plt.plot(x,fplot)
plt.plot(x,y)
plt.scatter(func_root,0,marker='o',color='blue')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlim(0,1.5)
plt.ylim(-1,1)   
    