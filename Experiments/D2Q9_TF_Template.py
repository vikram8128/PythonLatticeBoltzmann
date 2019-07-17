# Copyright (C) Vikram Singh 2019
# Email: vikramsingh8128@gmail.com

'''Lines marked with a triple ### should be used to alter the conditions of the simulation'''

import tensorflow as tf
import numpy as np
import sys
import os
import cv2

cur_dir = os.getcwd()
sess = tf.compat.v1.InteractiveSession()

timeSteps = 200
stepsPerFrame = 100
Re = 10.0 ### Reynolds number (adjust for viscosity)

nx = 420 ### Lattice Dimensions
ny = 180

### Constants for obstacle (Example for flow around a cylinder)
cx = nx//4
cy = ny//2
r = ny//9

l = r ### Characteristic length

uLB = 0.04 ### Speed of fluid flow in lattice units


relax = 1/(3*(uLB*tf.dtypes.cast(l,tf.dtypes.float32)/Re)+0.5)


def inObstacle(x,y) : ### Boolean function for obstacle (Example for flow around a cylinder)
    return tf.dtypes.cast((x-cx)**2 + (y-cy)**2 < r**2,tf.dtypes.float32)

### Replace inObstacle(x,y) with the following to get obstacle from png image (black is obstacle, white is fluid)

# img_path = 'ObstacleProfiles/aerofoil.png'
# img = cv2.imread(img_path, 0)
# ny,nx = img.shape

# def inObstacle(x,y) :
#     return tf.dtypes.cast(img[[[int(a) for a in ly] for ly in y],[[int(b) for b in lx] for lx in x]] < 128, tf.dtypes.float32)


def iniVel(x,y,d) : ### Function describing inflow velocities at position (x,y), with direction d (d=0 is x-component, d=1 is y-component) (Example for flow around a cylinder)
    return (1-d)*(uLB)*(1 + 1e-4*np.sin(y/(ny-1)*2*np.pi))


### Uncomment whichever outflowFin function required
# def outflowFin(fin) : # Outflow function for left edge outflow
#     finL = tf.unstack(fin,axis=0)
#     finLb1 = tf.unstack(finL[0],axis=1)
#     finLb2 = tf.unstack(finL[1],axis=1)
#     finLb1[0] = finLb2[0]
#     finLb1[1] = finLb2[1]
#     finLb1[2] = finLb2[2]
#     finL[0] = tf.stack(finLb1,axis=1)
#     finL[1] = tf.stack(finLb2,axis=1)
#     fin = tf.stack(finL,axis=0)
#     return fin

def outflowFin(fin) : # Outflow function for right edge outflow
    finL = tf.unstack(fin,axis=0)
    finLb1 = tf.unstack(finL[-1],axis=1)
    finLb2 = tf.unstack(finL[-2],axis=1)
    finLb1[6] = finLb2[6]
    finLb1[7] = finLb2[7]
    finLb1[8] = finLb2[8]
    finL[-1] = tf.stack(finLb1,axis=1)
    finL[-2] = tf.stack(finLb2,axis=1)
    fin = tf.stack(finL,axis=0)
    return fin

# def outflowFin(fin) : # Outflow function for top edge outflow
#     finL = tf.unstack(fin,axis=1)
#     finLb1 = tf.unstack(finL[0],axis=1)
#     finLb2 = tf.unstack(finL[1],axis=1)
#     finLb1[2] = finLb2[2]
#     finLb1[5] = finLb2[5]
#     finLb1[8] = finLb2[8]
#     finL[0] = tf.stack(finLb1,axis=1)
#     finL[1] = tf.stack(finLb2,axis=1)
#     fin = tf.stack(finL,axis=1)
#     return fin

# def outflowFin(fin) : # Outflow function for bottom edge outflow
#     finL = tf.unstack(fin,axis=1)
#     finLb1 = tf.unstack(finL[-1],axis=1)
#     finLb2 = tf.unstack(finL[-2],axis=1)
#     finLb1[0] = finLb2[0]
#     finLb1[3] = finLb2[3]
#     finLb1[6] = finLb2[6]
#     finL[-1] = tf.stack(finLb1,axis=1)
#     finL[-2] = tf.stack(finLb2,axis=1)
#     fin = tf.stack(finL,axis=1)
#     return fin

col0 = tf.convert_to_tensor([0,1,2])
col1 = tf.convert_to_tensor([3,4,5])
col2 = tf.convert_to_tensor([6,7,8])

row2 = tf.convert_to_tensor([0,3,6])
row1 = tf.convert_to_tensor([1,4,7])
row0 = tf.convert_to_tensor([2,5,8])

### Uncomment whichever set of inflow functions (rho0, inFin) required
def rho0(fin,u) : # Inflow density function for left edge inflow
    r = tf.zeros((ny))
    for i in range(3) :
        r += ((fin[0,:,col1[i]]) + 2*(fin[0,:,col2[i]])) / (1-u[0,:,0])
    return r

def inFin0(fin,eq) : # Inflow population functions for left edge inflow
    return eq[0,:,0] + fin[0,:,8] - eq[0,:,8]
def inFin1(fin,eq) :
    return eq[0,:,1] + fin[0,:,7] - eq[0,:,7]
def inFin2(fin,eq) :
    return eq[0,:,2] + fin[0,:,6] - eq[0,:,6]



# def rho0(fin,u) : # Inflow density function for right edge inflow
#     r = tf.zeros((ny))
#     for i in range(3) :
#         r += ((fin[-1,:,col1[i]]) + 2*(fin[-1,:,col0[i]])) / (1+u[-1,:,0])
#     return r

# def inFin6(fin,eq) : # Inflow population functions for right edge inflow
#     return eq[-1,:,6] + fin[-1,:,2] - eq[-1,:,2]
# def inFin7(fin,eq) :
#     return eq[-1,:,7] + fin[-1,:,1] - eq[-1,:,1]
# def inFin8(fin,eq) :
#     return eq[-1,:,8] + fin[-1,:,0] - eq[-1,:,0]



# def rho0(fin,u) : # Inflow density function for top edge inflow
#     r = tf.zeros((nx))
#     for i in range(3) :
#         r += ((fin[:,0,row1[i]]) + 2*(fin[:,0,row2[i]])) / (1+u[:,0,1])
#     return r
    
# def inFin2(fin,eq) : # Inflow population functions for top edge inflow
#     return eq[:,0,2] + fin[:,0,6] - eq[:,0,6]
# def inFin5(fin,eq) :
#     return eq[:,0,5] + fin[:,0,3] - eq[:,0,3]
# def inFin8(fin,eq) :
#     return eq[:,0,8] + fin[:,0,0] - eq[:,0,0]



# def rho0(fin,u) : # Inflow density function for bottom edge inflow
#     r = tf.zeros((nx))
#     for i in range(3) :
#         r += ((fin[:,-1,row1[i]]) + 2*(fin[:,-1,row0[i]])) / (1-u[:,-1,1])
#     return r
    
# def inFin0(fin,eq) : # Inflow population functions for top edge inflow
#     return eq[:,-1,0] + fin[:,-1,8] - eq[:,-1,8]
# def inFin3(fin,eq) :
#     return eq[:,-1,3] + fin[:,-1,5] - eq[:,-1,5]
# def inFin6(fin,eq) :
#     return eq[:,-1,6] + fin[:,-1,2] - eq[:,-1,2]



qDim = 1 # Number of Lattice rings

def genV() : # Generator for lattice directions
    v = []
    for x in range(qDim,-qDim-1,-1) :
        for y in range(qDim,-qDim-1,-1) :
            v.append(tf.convert_to_tensor([tf.dtypes.cast(x,tf.float32),tf.dtypes.cast(y,tf.float32)]))
    return v

v = tf.convert_to_tensor(genV()) # Lattice direction vectors

t = tf.convert_to_tensor([1/36,1/9,1/36,1/9,4/9,1/9,1/36,1/9,1/36]) # Lattice weights



def simple_conv(x, k, strides): # Convolution function for velocities and imaging
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  k = tf.expand_dims(tf.expand_dims(k, -1), -1)
  y = tf.nn.conv3d(x, k, strides, padding='SAME')
  return y[0, :, :, :, 0]

def macroDense(fin) : # Macroscopic density
    return tf.math.reduce_sum(fin,axis=2)

def macroU(fin, rho) : # Macroscopic velocity
    u0 = simple_conv(fin,tf.expand_dims(tf.expand_dims(v[:,0],0),0),[1,1,1,9,1])[:,:,0]
    u1 = simple_conv(fin,tf.expand_dims(tf.expand_dims(v[:,1],0),0),[1,1,1,9,1])[:,:,0]
    u0 /= rho
    u1 /= rho
    return tf.stack([u0,u1],axis=2)

def equilibrium(rho,u) : # Macroscopic equilibrium populations
    usqr = (3/2 * (u[:,:,0]**2 + u[:,:,1]**2))
    eqs = []
    for i in range(9) :
        cu = 3*(v[i,0]*u[:,:,0] + v[i,1]*u[:,:,1])
        eqs.append(rho*t[i]*(1+cu+0.5*cu**2-usqr))
    return tf.stack(eqs,axis=2)

def getImage(u) : # Translation function from velocities to tensor for imaging
    usq =  u**2
    return (2**3)*tf.math.sqrt(simple_conv(usq,tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([tf.dtypes.cast(1.0,tf.float32),tf.dtypes.cast(1.0,tf.float32)]),0),0),[1,1,1,2,1]))


obstacle = np.fromfunction(inObstacle,(nx,ny)) #generator for obstacle tensor


def obstacleConv(f) : # Helper function for creating obstacle mask
    ret = []
    for i in range(9) :
        ret.append(f[:,:,8-i])
    return tf.stack(ret,axis=2)

def obstacleRef(fin, fout, obstacle) : # Implements non-slip boundary on obstacle
    a = fout*(1-tf.stack([obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle],axis=2))
    return a + obstacleConv(tf.math.multiply(fin,tf.stack([obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle],axis=2)))

def collideOut(fin,eq) : # Collision operator
    return fin-relax*(fin-eq)

def streamRoller(fout) : # Streaming operator
    ret = []
    for i in range(9) :
        ret.append(tf.roll(tf.roll(fout[:,:,i], tf.dtypes.cast(v[i,0],tf.dtypes.int32), axis=0),tf.dtypes.cast(v[i,1],tf.dtypes.int32), axis=1))
    return tf.stack(ret,axis=2)

vel = tf.cast(tf.convert_to_tensor(np.fromfunction(iniVel,(nx,ny,2))),tf.float32) # creates initial velocity conditions

fininit = tf.zeros((nx,ny,9), dtype=tf.float32)
fininit = equilibrium(1,vel) # Creates initial populations

fin = tf.Variable(fininit) # Populations flowing into each point
fout = tf.Variable(tf.zeros((nx,ny,9), dtype=tf.float32)) # Populations flowing out of each point


u = tf.Variable(tf.zeros((nx,ny,2), dtype=tf.float32)) # Macroscopic velocity
rho = tf.Variable(tf.zeros((nx,ny), dtype=tf.float32)) # Macroscopic density
eq = tf.Variable(tf.zeros((nx,ny,9), dtype=tf.float32)) # Macroscopic equilibrium populations



step01 = tf.compat.v1.assign(fin,outflowFin(fin)) # Population outflow

with tf.compat.v1.get_default_graph().control_dependencies([step01]):
    step02 = tf.compat.v1.assign(rho,macroDense(fin)) # Macroscopic density calculation

    with tf.compat.v1.get_default_graph().control_dependencies([step02]):
        step03 = tf.compat.v1.assign(u,macroU(fin,rho)) # Macroscopic velocity calculation

        with tf.compat.v1.get_default_graph().control_dependencies([step03]): # Inflow velocity
            ### Uncomment whichever inflow velocity statement required
            step04 = tf.compat.v1.assign(u[0,:,:],vel[0,:,:]) # For left edge inflow
            # step04 = tf.compat.v1.assign(u[-1,:,:],vel[-1,:,:]) # For right edge inflow
            # step04 = tf.compat.v1.assign(u[:,0,:],vel[:,0,:]) # For top edge inflow
            # step04 = tf.compat.v1.assign(u[:,-1,:],vel[:,-1,:]) # For bottom edge inflow

            with tf.compat.v1.get_default_graph().control_dependencies([step04]): #Inflow Density
                ### Uncomment whichever inflow density statement required
                step05 = tf.compat.v1.assign(rho[0,:],rho0(fin,u)) # For left edge inflow
                # step05 = tf.compat.v1.assign(rho[-1,:],rho0(fin,u)) # For right edge inflow
                # step05 = tf.compat.v1.assign(rho[:,0],rho0(fin,u)) # For top edge inflow
                # step05 = tf.compat.v1.assign(rho[:,-1],rho0(fin,u)) # For bottom edge inflow
                        
                with tf.compat.v1.get_default_graph().control_dependencies([step05]):
                    step06 = tf.compat.v1.assign(eq,equilibrium(rho,u)) # Macroscopic equilibrium population calculation

                    with tf.compat.v1.get_default_graph().control_dependencies([step06]): # Inflow population calculation
                        ### Uncomment whichever inflow population set of statements required
                        step07 = tf.compat.v1.assign(fin[0,:,0],inFin0(fin,eq)) #For left edge inflow
                        step08 = tf.compat.v1.assign(fin[0,:,1],inFin1(fin,eq))
                        step09 = tf.compat.v1.assign(fin[0,:,2],inFin2(fin,eq))

                        
                        # step07 = tf.compat.v1.assign(fin[-1,:,6],inFin6(fin,eq)) # For right edge inflow
                        # step08 = tf.compat.v1.assign(fin[-1,:,7],inFin7(fin,eq))
                        # step09 = tf.compat.v1.assign(fin[-1,:,8],inFin8(fin,eq))

                        
                        # step07 = tf.compat.v1.assign(fin[:,0,2],inFin2(fin,eq)) # For top edge inflow
                        # step08 = tf.compat.v1.assign(fin[:,0,5],inFin5(fin,eq))
                        # step09 = tf.compat.v1.assign(fin[:,0,8],inFin8(fin,eq))

                        
                        # step07 = tf.compat.v1.assign(fin[:,-1,0],inFin0(fin,eq)) # For bottom edge inflow
                        # step08 = tf.compat.v1.assign(fin[:,-1,3],inFin3(fin,eq))
                        # step09 = tf.compat.v1.assign(fin[:,-1,6],inFin6(fin,eq))

                        with tf.compat.v1.get_default_graph().control_dependencies([step07,step08,step09]):
                            step10 = tf.compat.v1.assign(fout,collideOut(fin,eq)) # Collision calculation

                            with tf.compat.v1.get_default_graph().control_dependencies([step10]):
                                step11 = tf.compat.v1.assign(fout,obstacleRef(fin,fout,obstacle)) # Obstacle collision calculation

                                with tf.compat.v1.get_default_graph().control_dependencies([step11]):
                                    step12 = tf.compat.v1.assign(fin,streamRoller(fout)) # Streaming calculation

                                    with tf.compat.v1.get_default_graph().control_dependencies([step12]):
                                        step13 = tf.compat.v1.assign(fin,fin) # Garbage line to force execution of streaming step

stepGroup = tf.group(step13,)

tf.compat.v1.global_variables_initializer().run()

for time in range(timeSteps*stepsPerFrame):
    sess.run(stepGroup)
    if (time%stepsPerFrame == 0) :
        img = tf.image.encode_png(tf.image.convert_image_dtype(getImage(u),tf.dtypes.uint16))
        f = open("output/vel.{0:04d}.png".format(time//stepsPerFrame), "wb+")
        f.write(img.eval())
        f.close()
        