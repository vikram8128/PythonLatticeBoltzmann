# Copyright (C) Vikram Singh 2019
# Email: vikramsingh8128@gmail.com

from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm



#Flow parameters
iterations = 20000
Re = 10.0 #Reynolds number
nx = 420
ny = 180
ly = ny-1 # domain height
cx, cy, r = nx//4, ny//2, ny//9 # location of obstacle
uLB = 0.04 # inflow velocity
nulb = (uLB*r/Re) # viscosity
relax = 1/(3*nulb+0.5) # relaxation parameter

#Lattice constants
qDim = 1 # number of rings in lattice direction grid

def genV() : # Generator for lattice directions
    v = []
    for x in range(qDim,-qDim-1,-1) :
        for y in range(qDim,-qDim-1,-1) :
            v.append([x,y])
    return v

v = array(genV())

t = array([1/36,1/9,1/36,1/9,4/9,1/9,1/36,1/9,1/36])

col1 = array([0,1,2])
col2 = array([3,4,5])
col3 = array([6,7,8])

def macroDense(fin) :
    rho = sum(fin,axis=0)
    u = zeros((2,nx,ny))
    for i in range(9) :
        u[0,:,:] += v[i,0]*fin[i,:,:]
        u[1,:,:] += v[i,1]*fin[i,:,:]
    u /= rho
    return rho,u

def equilibrium(rho,u) :
    usqr = 3/2 * (u[0]**2 + u[1]**2)
    eq = zeros((9,nx,ny))
    for i in range(9) :
        cu = 3*(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        eq[i,:,:] = rho*t[i]*(1+cu+0.5*cu**2-usqr)
    return eq

# Create Obstacle
def inObstacle(x,y) :
    return (x-cx)**2 + (y-cy)**2 < r**2

obstacle = fromfunction(inObstacle,(nx,ny))

#Initial velocity (slight disturbance to give instability)

def iniVel(d,x,y) :
    return (1-d)*uLB*(1 + 1e-4*sin(y/ly*2*pi))

vel = fromfunction(iniVel,(2,nx,ny))

fin = equilibrium(1,vel)

for time in range(iterations) :
    fin[col3,-1,:] = fin[col3,-2,:] # outflow
    
    rho,u = macroDense(fin)
    #inflow
    u[:,0,:] = vel[:,0,:]
    rho[0,:] = ((sum(fin[col2,0,:],axis=0)) + 2*(sum(fin[col3,0,:],axis=0))) / (1-u[0,0,:])

    #equlibrium
    eq = equilibrium(rho,u)
    fin[[0,1,2],0,:] = eq[[0,1,2],0,:] + fin[[8,7,6],0,:] - eq[[8,7,6],0,:]

    #collision
    fout = fin-relax*(fin-eq) 

    #obstacle
    for i in range(9):
        fout[i,obstacle] = fin[8-i,obstacle]

    #streaming
    for i in range(9):
        fin[i,:,:] = roll(roll(fout[i,:,:], v[i,0], axis=0),v[i,1], axis=1)

    if(time%100==0) :
        plt.clf()
        plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(),cmap=cm.Blues)
        plt.savefig("vel.{0:04d}.png".format(time//100))