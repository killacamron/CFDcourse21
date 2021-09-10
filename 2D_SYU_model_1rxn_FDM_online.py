# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy import genfromtxt
from pathlib import Path
from datetime import datetime as date
import math
import ruptures as rpt

plt.rcParams['interactive'] == True

j = 0
samp_int = 60.0                                 # this term is the sampling interval (frequency that model solves)
check_change = 0
D = 0.0015875                                   # tubing diameter in m
zl = 30/100                                     # tubing length in m & x range
rl = D/2                                          # tubing diameter & y range
nz = 300                                        # x grid points
nr = 50                                         # y grid points
dz = zl/(nz-1)                                  # x stepsize
dr = rl/(nr-1)                                  # y stepsize
D1 = 1e-8 # axial dispersion coefficient (m^2/s)
k= .12                                          # thermal conductvity W/(m*K)
p = 1750                                        # density (kg/m3)
Cp = 1172                                       # specifc heat (J/kg/K)
a = k/(p*Cp)                                    # thermal diffusivity (m2/s)
dt = .0005
Ac = math.pi*(D/2)**2                           # cross-sectional area (m2)
Vr = Ac*zl                        # tubing volume (m3)
lamz = a*dt/dz**2                               # lumped coefficient
lamr = a*dt/dr**2                               # lumped coefficient
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
A0 = 1200 # stock concentration of species A (acrylate) (mol/m3)
B0 = 1000 # stock concentration of species B (fluoro) (mol/m3)

centerline = nr/2
wallline = nr-5
centerline = int(centerline)
wallline = int(wallline)

# grid formation
z = np.linspace(0, zl, nz) 
r = np.linspace(0, rl, nr) 
Z, R = np.meshgrid(z, r) 

# function that reads in previously collected .csv data for flowrates/temperature 
def data_reading_online():
    global data_reading_online
    global Jkem_folder, Jkem_file,Jkem_raw,temp_time,temp,dft,SMA_temp,Path
    global acrylate_raw,acrylate_time,acrylate_flow,df,SMA_acrylate_flow
    global fluoro_raw,fluoro_time,fluoro_flow,df2,SMA_fluoro_flow
    global cyclo_raw,cyclo_time,cyclo_flow,df3,SMA_cyclo_flow
    global counter,sol_array
    
    Jkem_raw= genfromtxt('7-03-temp-CA2-183_correctset.csv',dtype='str',delimiter=',',invalid_raise=False,skip_header=4) 
    temp_time = Jkem_raw[0:len(Jkem_raw),0]
    temp = Jkem_raw[0:len(Jkem_raw),1]
    temp = temp.astype('float')
    dft = pd.DataFrame({'temp_time':temp_time,'temp':temp})
    dft['SMA_temp'] = dft.iloc[:,1].rolling(window=60,min_periods=1).mean()
    
    # acrylate (species A) data is read in from .csv and dataframe created for time and flowrate values and moving average
    # data must be read in as strings, quotes stripped away, then flowrate converted to float
    acrylate_raw = genfromtxt('acrylate-7-03-CA2-183.csv',delimiter=',',invalid_raise=False,skip_header=17,dtype='str',usecols=(2,3))
    acrylate_raw = np.char.strip(acrylate_raw,chars='"')
    acrylate_time = acrylate_raw[0:len(acrylate_raw),0]
    acrylate_flow = acrylate_raw[0:len(acrylate_raw),1]
    acrylate_flow = acrylate_flow.astype(float)
    df = pd.DataFrame({'acrylate_time':acrylate_time,'acrylate_flow':acrylate_flow})
    df['acrylate_time'] = pd.to_datetime(df.acrylate_time,format= '%H:%M:%S.%f')
    df['acrylate_time'] = df['acrylate_time'].astype(str)
    df['acrylate_time'] = (df['acrylate_time'].str.split('01-01').str[1].astype(str))
    df['acrylate_time'] = (df['acrylate_time'].str.split('.').str[-2].astype(str))
    df['SMA_acrylate_flow'] = df.iloc[:,1].rolling(window=60,min_periods=1).mean()
    
    # fluoro (species B) data is read in from .csv and dataframe created for time and flowrate values and moving average
    # data must be read in as strings, quotes stripped away, then flowrate converted to float
    fluoro_raw = genfromtxt('fluoro-7-03-CA2-183.csv',delimiter=',',invalid_raise=False,skip_header=17,dtype='str',usecols=(2,3))
    fluoro_raw = np.char.strip(fluoro_raw,chars='"')
    fluoro_time = fluoro_raw[0:len(fluoro_raw),0]
    fluoro_flow = fluoro_raw[0:len(fluoro_raw),1]
    fluoro_flow = fluoro_flow.astype(float)
    df2 = pd.DataFrame({'fluoro_time':fluoro_time,'fluoro_flow':fluoro_flow})
    df2['fluoro_time'] = pd.to_datetime(df2.fluoro_time,format= '%H:%M:%S.%f')
    df2['fluoro_time'] = df2['fluoro_time'].astype(str)
    df2['fluoro_time'] = (df2['fluoro_time'].str.split('01-01').str[1].astype(str))
    df2['fluoro_time'] = (df2['fluoro_time'].str.split('.').str[-2].astype(str))
    df2['SMA_fluoro_flow'] = df2.iloc[:,1].rolling(window=60,min_periods=1).mean()


# function that initializes arrays for solver and to temporarily store solutions
# acting as initial conditions for each species
def SYU_Initialize_Arrays_online():
    global SYU_Arrays_online
    global A,B,C,An,Bn,Cn
    global C2,C2m,D,Dm,E,Em, dfR1sol, dfR2sol, CT, ET, tolcheck, T, Tn
    global chnghist_f
    global chnghist_a
    global chnghist_T
    for i in range(1):
        # R1 solutions arrays
        T = np.ones((nr, nz))  
        Tn = np.ones((nr, nz)) 
        A = np.ones((nr, nz))  
        An = np.ones((nr, nz)) 
        B = np.ones((nr, nz))  
        Bn = np.ones((nr, nz)) 
        C = np.ones((nr, nz))  
        Cn = np.ones((nr, nz)) 
        tolcheck = np.zeros((nr,nz))
        
        dfR1sol = pd.DataFrame(columns=['R1_Conc','R1_time'])
        dfR1sol = dfR1sol.append({'R1_Conc':0,'R1_time':0},ignore_index=True)
        dfR1sol['SMA_R1_Conc'] = dfR1sol.iloc[:,0].rolling(window=3,min_periods=1).mean()
        
        dfc = pd.DataFrame(columns=['calibrated mL/min','time'])
        dfc = dfc.append({'calibrated mL/min':0,'time':0},ignore_index=True)
        df2c = pd.DataFrame(columns=['calibrated mL/min','time'])
        df2c = df2c.append({'calibrated mL/min':0,'time':0},ignore_index=True)
        df4c = pd.DataFrame(columns=['temp','time'])
        df4c = df4c.append({'temp':0,'time':0},ignore_index=True)
        
        chnghist_f = pd.DataFrame(columns=['f_changetime'])
        chnghist_a = pd.DataFrame(columns=['a_changetime'])
        chnghist_T = pd.DataFrame(columns=['T_changetime'])
        break

def SYU_Arrays_updateIC_online():
    global SYU_Arrays_updateIC_online
    global A,B,C,An,Bn,Cn
    global C2,C2m,D,Dm,E,Em,CT,ET, dfR2sol, T, Tn, tolcheck
    for i in range(1):
      if j == 1:
        break
      else:
        # R1 solutions arrays
        A = A.copy() # species A solution array 
        An = A.copy() # species A temporary solution array
        B = B.copy() # species B solution array
        Bn = B.copy() # species B temporary solution array
        C = C.copy() # species C solution array
        Cn = C.copy() # species C temporary solution array
        T = T.copy()
        Tn = T.copy()
        tolcheck = np.zeros((nr,nz))
        break    
    
# function that uses flowrate readings to calculate combined flowrates and velocity
# flowrates from sensirion are uL/s, converted to mL/min with calibration, then unit conversion to m3/s
def SYU_flowrates_online():
    global SYU_flowrates_online
    global Q1,SMA_f,SMA_a,u,F_f,F_a
    global Q2, SMA_c,F_c,u2,F_R1
    
    # R1 flowrates in realtime
    SMA_f = ((df2.SMA_fluoro_flow[len(fluoro_raw)-1]*0.00387)-0.273)*1.66667*10**-8 # current moving average flowrate flouro (species B) (m3/s)
    SMA_a = ((df.SMA_acrylate_flow[len(acrylate_raw)-1]*0.00897)-.0769)*1.66667*10**-8 # current moving average flowrate acrylate (species A) (m3/s)
    Q = SMA_f + SMA_a # first reactor combined volumetric flowrate (m3/s)
    uAvg = Q/Ac                                     # average velocity (m/s)
    uMax = 2*uAvg                                   # max velocity (m/s)
    u = np.zeros(nr)                                # array initilization
    u[:] = np.linspace(0,rl,nr)                     # array intialization
    u[:] = uMax*(1-(u[:]/(D/2))**2)                 # hagan-poiselle profile
    u[-1] = 0                                       # no slip BC
    u = np.array([u,]*nz)                           # velocity field
    u = u.T                                         # transpose/align field
    F_f = B0*SMA_f/Q # fluoro (specis B) molar stream concentration (mol/m3)
    F_a = A0*SMA_a/Q # acrylate (specis A) molar stream concentration (mol/m3)
    
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
  
def impose_2D_BC_Conc(var,val1):
    var[-1,:] = var[-2,:] # tubing wall temp dirichlet BC
    var[0, :] = var[1,:] #symmetry condition
    var[:, 0] = val1 # inlet flow temp dirichlet BC
    var[:, -1] = var[:,-2] # outlet flow temp neumann BC
    return var

# function that uses all defined and calculated values to solve for the reaction progression in the tubular reactor
def SYU_solver_vector_online():
    global SYU_solver_vector_online, CT, ET, Cn, Em, dfR1sol, F_R1, tolcheck
    global T, Tn, impose_2D_BC_T, impose_2D_BC_Conc,F_f, F_a, p, Cp, h, zl, Vr, Tw
    
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
        A[:] = impose_2D_BC_Conc(A,F_f)
        B[:] = impose_2D_BC_Conc(B,F_a)
        C[:] = impose_2D_BC_Conc(C,0.0)
        
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
    print(stepcount)


def SYU_visualize_online():
    plt.ion()
    plt.figure(1,figsize=(15,15))
    plt.clf()
    fig1 = plt.subplot(221)
    cont = plt.contourf(Z,R,T[:]-273.15,50)
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
    
    fig2 = plt.subplot(222)
    plt.plot(z, centerT,label='center')
    plt.plot(z,wallT,label='wall')
    plt.legend()
    plt.ylabel('Temperature (degC)')
    plt.xlabel('Tubing Length (m)')
    
    fig3 = plt.subplot(223)
#    ax = fig1.gca()
#    plt.imshow(T[:])
    cont = plt.contourf(Z,R,C[:]/1000,50)
    ax = plt.gca()
    #ax.axis('scaled')
    ax.axes.get_yaxis().set_visible(True)
    plt.xlim(0,zl)
    plt.yticks([0,D/2],['Center (r = 0)','Wall (r = R)'],fontsize ='10')
    plt.xlabel('Tubing Length (m)')
    cbar = plt.colorbar(cont)
    cbar.ax.set_ylabel('Concentration (M)')
    
    fig4 = plt.subplot(224)
    ax=plt.gca()
    dfR1sol.plot(x='R1_time',ax=ax,y='R1_Conc',legend=None)
    plt.ylabel('Concentration (M)')
    plt.ylim([0,0.5])
    plt.xlabel('time (HH:MM:SS)')
    plt.xticks(dfR1sol.index,rotation=45)
    plt.xlim([j-10,j])

    plt.savefig("count_%s.jpg"%(j))
    plt.pause(0.1)
    plt.draw

def changepoint_detector():
  global chnghist_f
  global chnghist_a
  global chnghist_T  
  model = 'rbf'
  flow_a = df.SMA_acrylate_flow[len(df.SMA_acrylate_flow)-300:]
  flow_a = flow_a.to_numpy()
  flow_f = df2.SMA_fluoro_flow[len(df2.SMA_fluoro_flow)-300:]
  flow_f = flow_f.to_numpy()
  temp_1 = dft.SMA_temp[len(dft.SMA_temp)-300:]
  temp_1 = temp_1.to_numpy()
  algo_a = rpt.Pelt(model=model).fit(flow_a)
  result_a = algo_a.predict(pen=15)
  algo_f = rpt.Pelt(model=model).fit(flow_f)
  result_f = algo_f.predict(pen=15)
  algo_T = rpt.Pelt(model=model).fit(temp_1)
  result_T = algo_T.predict(pen=4)
  if len(result_a) !=0:
      len_a = len(result_a)
      for k in range(len_a):
          chnghist_a = chnghist_a.append({'a_changetime':df.acrylate_time[result_a[k]]},ignore_index=True)
      print('Changepoint detected in acrylate flowrate')
  if len(result_f) !=0:
      len_f = len(result_f)
      for k in range(len_f):
          chnghist_f = chnghist_f.append({'f_changetime':df2.fluoro_time[result_f[:]]},ignore_index=True)
      print('Changepoint detected in fluoro flowrate')    
  if len(result_T) !=0:
      len_T = len(result_T)
      for k in range(len_T):
          chnghist_T = chnghist_T.append({'T_changetime':dft.temp_time[result_T[:]]},ignore_index=True)
      print('Changepoint detected in reactor 1 temperature')
  # rpt.display(temp_1, result_T, figsize=(10, 6))
  # plt.show() 
  # rpt.display(flow_f, result_f, figsize=(10, 6))
  # plt.show() 
  # rpt.display(flow_a, result_a, figsize=(10, 6))
  # plt.show() 
  chnghist_f.to_csv('fluoro_changelog.csv', index =False)
  chnghist_a.to_csv('acrylate_changelog.csv', index =False)
  chnghist_T.to_csv('temp_changelog.csv', index =False)
  
  
  
# model update loop, runs continuously until manually stopped, or if data files are not present to read
plt.ion()
starttime = date.now()
SYU_Initialize_Arrays_online() # calls function to re-initializes arrays
while True:    
    j += 1
    init = date.now()
    data_reading_online() # reads in and updates data into dataframes
    SYU_Arrays_updateIC_online() # calls function to re-initializes arrays
    SYU_flowrates_online() # calls flowrate function
    SYU_solver_vector_online() # calls solver
    centerT = T[centerline,:]
    wallT = T[wallline,:]
    centerC = C[centerline,:]
    wallC = C[wallline,:]
    outletC = np.average(C[:,-1])
    dfR1sol = dfR1sol.append({'R1_Conc':outletC/1000,'R1_time':fluoro_time[len(fluoro_time)-1]},ignore_index=True)
    dfR1sol['SMA_R1_Conc'] = dfR1sol.iloc[:,0].rolling(window=3,min_periods=1).mean()
    dfR1sol.to_csv('testing.csv',index=False)
    SYU_visualize_online() # calls visualization function
    if check_change == 1:
      changepoint_detector()
    end = date.now()
    runtime = end - starttime
    runtime = runtime.total_seconds()
    if runtime > 300.0:
      check_change == 1
    cputime = end-init
    cputime = cputime.total_seconds()
    print(cputime)
    #wait = samp_int - cputime
    wait = 1.1*cputime
    if wait <= 0.0:
      wait = 0.0
    time.sleep(wait) # pauses model loop before refreshing and updating next solution
    

      