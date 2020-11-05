# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:42:09 2020
Molecular Dynamics - FCC lattice and Lennard Jones Potential
@author: Shreya Verma
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from array import *

"""
=======================================================================
======================Taking Parameters as Input=========================
=======================================================================
"""

L=1.0                                                                                                                                      #Reduced box length
nc=int(input("Enter the number of unit cells for cubic lattice (Use 3) :" ))
tts=int(input("Enter the number of time steps to be simulated (Use 100000) :" ))
Natom=4*(nc**3)                                                                                                                # total number of atoms
cell=1.0/nc                                                                                                                            #length of each unit cell
cell2=0.5*cell                                                                                                                        #half the unit cell length
Boxl=Natom**(1/3)                                                                                                             #length of box
delt=0.005                                                                                                                            #time step - delta t

"""
=======================================================================
======================Initial Position Generation==========================
=======================================================================
"""

rx=np.zeros((Natom), dtype=float)                                                                                   #arrays for positions            
ry=np.zeros((Natom), dtype=float)
rz=np.zeros((Natom), dtype=float)

rx[0]=0.0; ry[0]=0.0; rz[0]=0.0 
rx[1]=cell2; ry[1]=cell2; rz[1]=0.0
rx[2]=0.0; ry[2]=cell2; rz[2]=cell2
rx[3]=cell2; ry[3]=0.0; rz[3]=cell2  
m=0                                                                                                                                      #generating initial coordinates in fcc lattice
for Iz in range(0,nc):
    for Iy in range(0,nc):
        for Ix in range(0,nc):
            for Iref in range(0,4):
                rx[Iref+m]=rx[Iref]+(cell*Ix)
                ry[Iref+m]=ry[Iref]+(cell*Iy)
                rz[Iref+m]=rz[Iref]+(cell*Iz)
                continue
            m=m+4
            continue
        continue
    continue
rx=rx*Boxl                                                                                                                                #Scaling positions
ry=ry*Boxl
rz=rz*Boxl

"""
=======================================================================
=====================Initial Velocity Generation===========================
=======================================================================
"""
np.random.seed(3)
vx=np.zeros((Natom), dtype=float)                                                                                       #arrays for velocities
vy=np.zeros((Natom), dtype=float)
vz=np.zeros((Natom), dtype=float)
temp=np.zeros((tts+1), dtype=float)                                                                                     #array for temperature at each MD step
sumvx=0.0
sumvy=0.0
sumvz=0.0
sumv2=0.0

for i in range(0,Natom):                                                                                                         #generating random initial velocities in [-0.5,0.5]
    vx[i]=(np.random.random())-0.5
    vy[i]=(np.random.random())-0.5
    vz[i]=(np.random.random())-0.5
    sumvx=sumvx+vx[i]
    sumvy=sumvy+vy[i]
    sumvz=sumvz+vz[i]
       
sumvx=sumvx/Natom                                                                                                            #centre of mass initially
sumvy=sumvy/Natom
sumvz=sumvz/Natom
    
for i in range(0,Natom):
    vx[i]=vx[i]-sumvx
    vy[i]=vy[i]-sumvy
    vz[i]=vz[i]-sumvz
    sumv2=sumv2+((vx[i]**2)+(vy[i]**2)+(vz[i]**2))
                                                                                                                                                     
df=(3*Natom)-3                                                                                                                       #calculating initial temperature before scaling
temp[0]=sumv2/df  
sf=np.sqrt(1/temp[0])                                                                                                             #scaling factor

sumv2=0.0                                                                                                                                #scaling the velocities
vx=vx*sf
vy=vy*sf
vz=vz*sf
for i in range(0,Natom):                                                                                                         #initial kinetic energy 
    sumv2=sumv2+((vx[i]**2)+(vy[i]**2)+(vz[i]**2))

"""
=======================================================================
=================Centre of Mass Velocity Calculation=======================
=======================================================================
"""

def vcom(vx,vy,vz):                                                                                                                  #function takes velocity arrays and returns com for a particular configuration/ MD step
    vcomx=0.0; vcomy=0.0; vcomz=0.0
    for i in range(Natom):
        vcomx=vcomx+vx[i]
        vcomy=vcomy+vy[i]
        vcomz=vcomz+vz[i]
    vcomx=vcomx/Natom
    vcomy=vcomy/Natom
    vcomz=vcomz/Natom
    return(vcomx,vcomy,vcomz)

"""
=======================================================================
=======================Kinetic Energy Calculation==========================
=======================================================================
"""

def kin_en(vx,vy,vz):                                                                                                                #function takes velocity arrays and returns twice kinetic energy for a particular configuration/ MD step
    EKin=0.0
    for i in range(Natom):
        EKin=EKin+vx[i]**2+vy[i]**2+vz[i]**2
    return(EKin)

"""
========================================================================
=========================Force Calculation================================
========================================================================
"""

fx=np.zeros((Natom), dtype=float)                                                                                       #arrays for forces
fy=np.zeros((Natom), dtype=float)
fz=np.zeros((Natom), dtype=float)
EP=np.zeros((tts+1), dtype=float)                                                                                         #arrays for energies
EK=np.zeros((tts+1), dtype=float)
TE=np.zeros((tts+1), dtype=float)
VCX=np.zeros((tts+1), dtype=float)                                                                                      #arrays for storing com velocities
VCY=np.zeros((tts+1), dtype=float)
VCZ=np.zeros((tts+1), dtype=float)
time=np.zeros((tts+1), dtype=float)                                                                                     #array for storing time
k=0
def force():                                                                                                                               #function to calculate forces and potential energies
    for i in range(Natom):
        fx[i]=0.0
        fy[i]=0.0
        fz[i]=0.0
    Epot=0.0
    rcsq=(2.5)**2                                                                                                                       #square cutoff radius
    for i in range(0,Natom-1):
        for j in range(i+1,Natom):
            rxij=rx[i]-rx[j]
            rxij=rxij-Boxl*round(rxij/Boxl)                                                                                    #pbc-x
            ryij=ry[i]-ry[j]
            ryij=ryij-Boxl*round(ryij/Boxl)                                                                                   #pbc-y
            rzij=rz[i]-rz[j] 
            rzij=rzij-Boxl*round(rzij/Boxl)                                                                                    #pbc-z
            rsq=(rxij*rxij)+(ryij*ryij)+(rzij*rzij)
            if (rsq <= rcsq):
                rsqinv=1/rsq
                rsqcubeinv=1/(rsq*rsq*rsq)
                ff=48.0*rsqinv*rsqcubeinv*(rsqcubeinv-0.5)
                fx[i]=fx[i]+ff*rxij                                                                                                      #force calculations
                fx[j]=fx[j]-ff*rxij
                fy[i]=fy[i]+ff*ryij
                fy[j]=fy[j]-ff*ryij
                fz[i]=fz[i]+ff*rzij
                fz[j]=fz[j]-ff*rzij
                Epot=Epot+4.0*rsqcubeinv*(rsqcubeinv-1.0)                                                    #calculating potential energy 
    EP[k]=Epot                                                                                                                          #storing potential energy for each MD step
    
    return()

"""
=======================================================================
===========Integration function to integrate equations of motion=============
=======================================================================
"""

def integrate():
    for i in range (0,Natom):
        rx[i]=(rx[i]+delt*vx[i]+0.5*delt*delt*fx[i])%Boxl                                                               #position updation at t+dt and PBCs in x,y,z
        ry[i]=(ry[i]+delt*vy[i]+0.5*delt*delt*fy[i])%Boxl                                                              
        rz[i]=(rz[i]+delt*vz[i]+0.5*delt*delt*fz[i])%Boxl                                                               
        vx[i]=vx[i]+0.5*delt*fx[i]                                                                                                      #velocity updation at t+dt/2
        vy[i]=vy[i]+0.5*delt*fy[i]
        vz[i]=vz[i]+0.5*delt*fz[i]
    
    force()                                                                                                                                        #calculating force for this instant
    
    for i in range (0,Natom):
        vx[i]=vx[i]+0.5*delt*fx[i]                                                                                                      #velocity updation at t+dt
        vy[i]=vy[i]+0.5*delt*fy[i]
        vz[i]=vz[i]+0.5*delt*fz[i]
    vcx,vcy,vcz=vcom(vx,vy,vz)                                                                                                      #calculate com velocity for this MD step
    VCX[k]=vcx; VCY[k]=vcy; VCZ[k]=vcz;
    EK[k]=kin_en(vx,vy,vz)/2                                                                                                          #store kinetic energy at this MD step
    temp[k]=kin_en(vx,vy,vz)/df                                                                                                    #store temperature at this MD step
    return

"""
=======================================================================
=========================Total Energy Calculation=========================
=======================================================================
"""

def total_en():                                                                                                                                 #function to calculate total energy, and time using scaling of time step with delta t 
    for i in range (0,tts+1):
        TE[i]=EK[i]+EP[i]
        time[i]=i*delt
    return
    
"""
=======================================================================
===========================Main  Program================================
=======================================================================
"""

k=0
force()                                                                                                                                              #call force to calculate initial force once
VCX[0]=sumvx; VCY[0]=sumvy; VCZ[0]=sumvz
EK[0]=sumv2/2

for t in range(0,tts):
    k=k+1
    integrate()                                                                                                                                   #call integrate function for each MD step
    t=t+delt
    print("Time ",t)                                                                                                                           #print time to check the run of program
total_en()                                                                                                                                         #call toal energy function

"""
==============================================================================
==============Writing data into Files and Other Results to be Reported===============
==============================================================================
"""

VCX_100=np.zeros((100), dtype=float)                                                                                        #arrays to store last 100 center of mass velocities, and time for printing in file  
VCY_100=np.zeros((100), dtype=float)
VCZ_100=np.zeros((100), dtype=float)
time_100=np.zeros((100), dtype=float)

for i in range(0,100):
    VCX_100[i]=VCX[tts-100+i+1]
    VCY_100[i]=VCY[tts-100+i+1]
    VCZ_100[i]=VCZ[tts-100+i+1]
    time_100[i]=time[tts-100+i+1]

velcom=np.vstack((time_100,VCX_100,VCY_100,VCZ_100))                                                     #writing last 100 steps' com velocities
velcom=velcom.T
np.savetxt('vel_com1.dat', velcom, delimiter='\t', header='X,Y,Z Components Centre of Mass Velocity at each MD step for last 100 steps')

Sum_EP=0.0; Sum_temp=0.0                            
for i in range(0,50000):                                                                                                                    #calculating average potential energy and temperature
    Sum_EP=Sum_EP+EP[tts-50000+i+1]
    Sum_temp=Sum_temp+temp[tts-50000+i+1]
Avg_EP=Sum_EP/50000
Avg_temp=Sum_temp/50000
print("Average Potential Energy for last 50000 steps : ",Avg_EP)
print("Average Temperature for last 50000 steps : ",Avg_temp)

"""
=======================================================================
================================Plots===================================
=======================================================================
"""

EP_f500=np.zeros((500), dtype=float)                                                                                            #Defining 6 subplots for all energies
EK_f500=np.zeros((500), dtype=float)
ET_f500=np.zeros((500), dtype=float)
EP_l500=np.zeros((500), dtype=float)
EK_l500=np.zeros((500), dtype=float)
ET_l500=np.zeros((500), dtype=float)
time_f500=np.zeros((500), dtype=float)
time_l500=np.zeros((500), dtype=float)

for i in range(0,500):                                                                                                                          #Storing Energy for First 500 steps
    EP_f500[i]=EP[i]
    EK_f500[i]=EK[i]
    ET_f500[i]=TE[i]
    time_f500[i]=time[i]

for i in range(0,500):                                                                                                                          #Storing Energy for Last 500 steps
    EP_l500[i]=EP[tts-500+i+1]
    EK_l500[i]=EK[tts-500+i+1]
    ET_l500[i]=TE[tts-500+i+1]
    time_l500[i]=time[tts-500+i+1]

"""
=======================================================================
Plotting with separate files for each of the 6 plots
=======================================================================
"""
plt.figure()
plt.plot(time_f500,EP_f500, color='g', label="Potential Energy")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc="upper right")
plt.title('Potential Energy for first 500 MD steps')
#plt.savefig('PE1.png', dpi=500)
plt.show()

plt.figure()
plt.plot(time_l500,EP_l500, color='g', label="Potential Energy")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc="upper right")
plt.title('Potential Energy for last 500 MD steps')
#plt.savefig('PE2.png', dpi=500)
plt.show()

plt.figure()
plt.plot(time_f500,EK_f500, color='b', label="Kinetic Energy")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc="upper right")
plt.title('Kinetic Energy for first 500 MD steps')
#plt.savefig('KE1.png', dpi=500)
plt.show()

plt.figure()
plt.plot(time_l500,EK_l500, color='b', label="Kinetic Energy")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc="upper right")
plt.title('Kinetic Energy for last 500 MD steps')
#plt.savefig('KE2.png', dpi=500)
plt.show()

plt.figure()
plt.plot(time_f500,ET_f500, color='r', label="Total Energy")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc="upper right")
plt.title('Total Energy for first 500 MD steps')
#plt.savefig('TE1.png', dpi=500)
plt.show()

plt.figure()
plt.plot(time_l500,ET_l500, color='r', label="Total Energy")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc="upper right")
plt.title('Total Energy for last 500 MD steps')
#plt.savefig('TE2.png', dpi=500)
plt.show()

"""
=======================================================================
==================================End==================================
=======================================================================
"""
