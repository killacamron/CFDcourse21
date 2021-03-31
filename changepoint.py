# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:24:55 2021

@author: Cam
"""

import changefinder
import numpy as np
from matplotlib import pyplot as plt
import ruptures as rpt
import datetime as dt

#example usage
# points=np.concatenate([np.random.rand(100)+5,
#                        np.random.rand(100)+20,
#                        np.random.rand(100)+5])

# f, (ax1, ax2) = plt.subplots(2, 1)
# f.subplots_adjust(hspace=0.4)
# ax1.plot(points)
# ax1.set_title("data point")
# #Initiate changefinder function
# cf = changefinder.ChangeFinder()
# scores = [cf.update(p) for p in points]
# ax2.plot(scores)
# ax2.set_title("anomaly score")
# plt.show() 

#181 data import
IR_raw_181 = np.genfromtxt('181_IR_data.csv',dtype='str',delimiter=',',skip_header=1)
IR_181 = IR_raw_181[64:359,1]
IR_181 = IR_181.astype('float')
IR_181_time = IR_raw_181[64:359,0]

R2_181_raw = np.genfromtxt('181testR2-3-17-21.csv',dtype='str',delimiter=',',skip_header=1)
R2_181_SMA = R2_181_raw[55:,2]
R2_181_SMA = R2_181_SMA.astype('float')
R2_181_SMA_time = R2_181_raw[55:,1]
R2_181_SMA_time = [x[:-5] for x in R2_181_SMA_time]
R2_181_SMA_time = np.array(R2_181_SMA_time)
R2_noSMA_181 = R2_181_raw[55:,0]
R2_noSMA_181 = R2_noSMA_181.astype('float')

#183 data import
fluoro_raw = np.genfromtxt('183test-fluoro.csv',dtype='str',delimiter=',',skip_header=1)
fluoro_flow = fluoro_raw[14:150,0]
fluoro_flow = fluoro_flow.astype('float')
fluoro_flow_time = fluoro_raw[14:150,1]
fluoro_flow_time = [x[:-5] for x in fluoro_flow_time]
fluoro_flow_time = np.array(fluoro_flow_time)


acrylate_raw = np.genfromtxt('183test-acrylate.csv',dtype='str',delimiter=',',skip_header=1)
acrylate_flow = acrylate_raw[14:150,0]
acrylate_flow = acrylate_flow.astype('float')
acrylate_flow_time = acrylate_raw[14:150,1]
acrylate_flow_time = [x[:-5] for x in acrylate_flow_time]
acrylate_flow_time = np.array(acrylate_flow_time)

cyclo_raw = np.genfromtxt('183test-cyclo.csv',dtype='str',delimiter=',',skip_header=1)
cyclo_flow = cyclo_raw[14:150,0]
cyclo_flow = cyclo_flow.astype('float')
cyclo_flow_time = cyclo_raw[14:150,1]
cyclo_flow_time = [x[:-5] for x in cyclo_flow_time]
cyclo_flow_time = np.array(cyclo_flow_time)


temp_raw = np.genfromtxt('183test-temp.csv',dtype='str',delimiter=',',skip_header=1)
temp = temp_raw[14:150,0]
temp=temp.astype('float')
temp_time = temp_raw[14:150,1]
#temp_time = [x[:-1] for x in temp_time]
temp_time = np.array(temp_time)


IR_raw_183 = np.genfromtxt('183_run1_IR.csv',dtype='str',delimiter=',',skip_header=2)
IR_183 = IR_raw_183[37:,1]
IR_183 = IR_183.astype('float')
IR_183_time = IR_raw_183[37:,0]

R2_raw_183 = np.genfromtxt('183testR2-3-17-21.csv',dtype='str',delimiter=',',skip_header=1)
R2_SMA_183 = R2_raw_183[14:150,2]
R2_SMA_183 = R2_SMA_183.astype('float')
R2_SMA_183_time = R2_raw_183[14:150,1]
R2_SMA_183_time = [x[:-5] for x in R2_SMA_183_time]
R2_SMA_183_time = np.array(R2_SMA_183_time)
R2_noSMA_183 = R2_raw_183[14:150,0]
R2_noSMA_183 = R2_noSMA_183.astype('float')

raw_resid_183 =  np.genfromtxt('183_residuals.csv',dtype='str',delimiter=',',skip_header=0)
resid = raw_resid_183[2:,1]
resid = resid.astype('float')
resid_time = raw_resid_183[2:,0]


plt.rcParams["font.family"] = "sans-serif"
plt.rc('axes',labelsize=18)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('figure',titlesize=10)


#181 using pelt method
model="rbf"
algo = rpt.Pelt(model=model).fit(IR_181)
result = algo.predict(pen=3)
rpt.display(IR_181, result, figsize=(10, 6))
xvalues = np.arange(0,len(IR_181),30)
tvalues = IR_181_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,110))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Concentration Species E (mg/mL)')
plt.title('Change Point Detection: IR Step Change Experiments')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(R2_181_SMA)
result = algo.predict(pen=3)
rpt.display(R2_181_SMA, result, figsize=(10, 6))
xvalues = np.arange(0,len(R2_181_SMA),30)
tvalues = R2_181_SMA_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,110))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Concentration Species E (mg/mL)')
plt.title('Change Point Detection: Dynamic Model Step Change Experiments')
plt.show() 


#183-1 using pelt method

model="rbf"
algo = rpt.Pelt(model=model).fit(fluoro_flow)
result = algo.predict(pen=3)
rpt.display(fluoro_flow, result, figsize=(10, 6))
xvalues = np.arange(0,len(fluoro_flow),15)
tvalues = fluoro_flow_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,3.5))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Flowrate Species B (mL/min)')
#plt.title('Change Point Detection: Pelt Search Method')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(acrylate_flow)
result = algo.predict(pen=3)
rpt.display(acrylate_flow, result, figsize=(10, 6))
xvalues = np.arange(0,len(acrylate_flow),15)
tvalues = acrylate_flow_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,3.5))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Flowrate Species A (mL/min)')
#plt.title('Change Point Detection: Pelt Search Method')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(cyclo_flow)
result = algo.predict(pen=3)
rpt.display(cyclo_flow, result, figsize=(10, 6))
xvalues = np.arange(0,len(cyclo_flow),15)
tvalues = cyclo_flow_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,3.5))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Flowrate Species D (mL/min)')
#plt.title('Change Point Detection: Pelt Search Method')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(temp)
result = algo.predict(pen=3)
rpt.display(temp, result, figsize=(10, 6))
xvalues = np.arange(0,len(temp),15)
tvalues = temp_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,160))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Reactor 1 Temperature (C)')
#plt.title('Change Point Detection: Pelt Search Method')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(IR_183)
result = algo.predict(pen=3)
rpt.display(IR_183, result, figsize=(10, 6))
xvalues = np.arange(0,len(IR_183),15)
tvalues = IR_183_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,110))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Concentration Species E (mg/mL)')
plt.title('Change Point Detection: IR Gradual Failure Experiments')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(R2_SMA_183)
result = algo.predict(pen=3)
rpt.display(R2_SMA_183, result, figsize=(10, 6))
xvalues = np.arange(0,len(R2_SMA_183),15)
tvalues = R2_SMA_183_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,110))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Concentration Species E (mg/mL)')
plt.title('Change Point Detection: Dynamic Model Gradual Failure Experiments')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(resid)
result = algo.predict(pen=3)
rpt.display(resid, result, figsize=(10, 6))
xvalues = np.arange(0,len(resid),15)
tvalues = resid_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Relative Residuals (-)')
plt.title('Change Point Detection: Residuals from IR & Dynamic Model Predictions')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(R2_noSMA_183)
result = algo.predict(pen=3)
rpt.display(R2_noSMA_183, result, figsize=(10, 6))
xvalues = np.arange(0,len(R2_SMA_183),15)
tvalues = R2_SMA_183_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,110))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Concentration Species E (mg/mL)')
plt.title('Change Point Detection: Dynamic Model Gradual Failure Experiments')
plt.show() 

model="rbf"
algo = rpt.Pelt(model=model).fit(R2_noSMA_181)
result = algo.predict(pen=3)
rpt.display(R2_noSMA_181, result, figsize=(10, 6))
xvalues = np.arange(0,len(R2_181_SMA),15)
tvalues = R2_181_SMA_time[xvalues]
plt.xticks(xvalues,tvalues)
plt.ylim((0,110))
plt.xlabel('time (HH:MM:SS)')
plt.ylabel('Concentration Species E (mg/mL)')
plt.title('Change Point Detection: Dynamic Model Gradual Failure Experiments')
plt.show() 
