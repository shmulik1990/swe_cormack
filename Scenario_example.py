# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:26:07 2022

@author: shmul
"""
#%% import libraries and functions from 'functions.py'
import numpy as np
import matplotlib.pyplot as plt
import sys

path=r'C:\Users\shmul\OneDrive\Documents\KIT_work\DATA_WORK\spyder_bks\swe'
sys.path.append(path)
import functions as F

#%% Scenario definition

# 1) define spatial domain

## horizontal
N=50 #number of computation segments
L=10     #length of hillslope in meters
x=np.linspace(1,L+1,num=N)
xp=np.linspace(1-(x[1]-x[0]),L+1+(x[-1]-x[-2]),num=N+2)   # incl. ghost points 

## vertical
### create Kirkby Soil Creep and Soil Wash slopes
xl=xp[-1]-xp[0]
xf=xp-xp[0]
YC={}
Ytyp=["Soil Creep","Rainsplash","Soil Wash","Rivers"]
ytyp=iter(Ytyp)
for mn in [(0.2,0.82),(1.0,1.11),(1.7,1.45),(2.5,1.97)]:
    m=mn[0]
    n=mn[1]
    y0=0.5
    nt=next(ytyp)
    YC[nt]=y0*(1-(xf/xl)**((1-m)/n+1))
    plt.plot(xf,YC[nt],label=mn)
plt.legend()

# 2) define time domain

tend=3600
pend=0.2*tend

# 3) define rainfall input

Peff=100/3600/1000
tss=np.linspace(0,tend,num=tend+1)
Pts=np.zeros(len(tss))
Pts[0:int(len(tss)*pend)+1]=Peff

# 4) define constants

g=9.81
beta=1
#%% Run code

c=r"C:\Users\shmul\\"
path_out=c+r"Documents"
name='test'

t,Q,H=F.SWE(oldrun=False,path_in=path_out,tname="tsave_"+name+"_n",Hname="Hsave_"+name+"_n",Qname="Qsave_"+name+"_n",name=name,x=x,
                        hini=.0001,tend=1200,ts=5,tss=tss,Pts=Pts,crstart=0.5,dtmax=0.01,
                        hgp=YC['Soil Creep'],n=0.1,hmin=0.0001,
                        save=True,path_out=path_out, rst=-99, plotQ=True)