# Copyright (C) Vikram Singh 2019
# Email: vikramsingh8128@gmail.com

'''Lines marked with a triple ### should be used to alter the conditions of the simulation'''

from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2


timeSteps = 200
stepsPerFrame = 100
Re = 10.0 ### Reynolds number (adjust for viscosity)

nx = 420 ### Lattice Dimensions
ny = 180

cx, cy, r = nx//4, ny//2, ny//9 ### Constants for obstacle (Example for flow around a cylinder)

uLB = 0.04 ### Speed of fluid flow in lattice units

l = r ### Characteristic length
nulb = (uLB*l/Re) # viscosity
relax = 1/(3*nulb+0.5) # relaxation parameter


def inObstacle(x,y) : ### Boolean function for obstacle (Example for flow around a cylinder)
    return (x-cx)**2 + (y-cy)**2 < r**2

### Replace inObstacle(x,y) with the following to get obstacle from png image (black is obstacle, white is fluid)

# img_path = 'ObstacleProfiles/airfoil.png'
# img = cv2.imread(img_path, 0)
# ny,nx = img.shape

# def inObstacle(x,y) :
#     return img[[[int(a) for a in ly] for ly in y],[[int(b) for b in lx] for lx in x]] < 128


def iniVel(d,x,y) : ### Function describing inflow velocities at position (x,y), with direction d (d=0 is x-component, d=1 is y-component) (Example for flow around a cylinder)
    return (1-d)*uLB*(1 + 1e-4*sin(y/(ny-1)*2*pi))



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

col0 = array([0,1,2])
col1 = array([3,4,5])
col2 = array([6,7,8])

row0 = array([2,5,8])
row1 = array([1,4,7])
row2 = array([0,3,6])

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


obstacle = fromfunction(inObstacle,(nx,ny))


vel = fromfunction(iniVel,(2,nx,ny))

fin = equilibrium(1,vel)

for time in range(timeSteps*stepsPerFrame) :


    ### Uncomment whichever outflow statement required
    # fin[col0,0,:] = fin[col0,1,:] # left edge outflow
    fin[col2,-1,:] = fin[col2,-2,:] # right edge outflow
    # fin[row0,:,0] = fin[row0,:,1] # top edge outflow
    # fin[row2,:,-1] = fin[row2,:,-2] # bottom edge outflow
    

    rho,u = macroDense(fin)


    ### Uncomment whichever set of inflow statements required
    u[:,0,:] = vel[:,0,:] # for left edge inflow
    rho[0,:] = ((sum(fin[col1,0,:],axis=0)) + 2*(sum(fin[col2,0,:],axis=0))) / (1-u[0,0,:])

    # u[:,-1,:] = vel[:,-1,:] # for right edge inflow
    # rho[-1,:] = ((sum(fin[col1,-1,:],axis=0)) + 2*(sum(fin[col0,-1,:],axis=0))) / (1+u[0,-1,:])

    # u[:,:,0] = vel[:,:,0] # for top edge inflow
    # rho[:,0] = ((sum(fin[row1,:,0],axis=0)) + 2*(sum(fin[row2,:,0],axis=0))) / (1+u[1,:,0])

    # u[:,:,-1] = vel[:,:,-1] # for bottom edge inflow
    # rho[:,-1] = ((sum(fin[row1,:,-1],axis=0)) + 2*(sum(fin[row0,:,-1],axis=0))) / (1-u[1,:,-1])



    #equlibrium
    eq = equilibrium(rho,u)
    fin[[0,1,2],0,:] = eq[[0,1,2],0,:] + fin[[8,7,6],0,:] - eq[[8,7,6],0,:] # for left edge inflow
    # fin[[6,7,8],-1,:] = eq[[6,7,8],-1,:] + fin[[2,1,0],-1,:] - eq[[2,1,0],-1,:] # for right edge inflow
    # fin[[2,5,8],:,0] = eq[[2,5,8],:,0] + fin[[6,3,0],:,0] - eq[[6,3,0],:,0] # for top edge inflow
    # fin[[0,3,6],:,-1] = eq[[0,3,6],:,-1] + fin[[8,5,2],:,-1] - eq[[8,5,2],:,-1] # for bottom edge inflow

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
        plt.savefig("output/vel.{0:04d}.png".format(time//stepsPerFrame))