# Copyright (C) Vikram Singh 2019
# Email: vikramsingh8128@gmail.com

import tensorflow as tf
import numpy as np
import sys
import os

cur_dir = os.getcwd()
sess = tf.compat.v1.InteractiveSession()

iterations = 200000
Re = 500.0
nx = 420
ny = 180
ly = ny-1
cx = nx//4
cy = ny//2
r = ny//9

uLB = 0.04
relax = 1/(3*(uLB*tf.dtypes.cast(r,tf.float32)/Re)+0.5)

qDim = 1 # number of rings in lattice direction grid

def genV() : # Generator for lattice directions
    v = []
    for x in range(qDim,-qDim-1,-1) :
        for y in range(qDim,-qDim-1,-1) :
            v.append(tf.convert_to_tensor([tf.dtypes.cast(x,tf.float32),tf.dtypes.cast(y,tf.float32)]))
    return v

v = tf.convert_to_tensor(genV())

t = tf.convert_to_tensor([1/36,1/9,1/36,1/9,4/9,1/9,1/36,1/9,1/36])

col1 = tf.convert_to_tensor([0,1,2])
col2 = tf.convert_to_tensor([3,4,5])
col3 = tf.convert_to_tensor([6,7,8])

def simple_conv(x, k, strides):
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  k = tf.expand_dims(tf.expand_dims(k, -1), -1)
  y = tf.nn.conv3d(x, k, strides, padding='SAME')
  return y[0, :, :, :, 0]

def macroDense(fin) :
    return tf.math.reduce_sum(fin,axis=2)

def macroU(fin, rho) :
    u0 = simple_conv(fin,tf.expand_dims(tf.expand_dims(v[:,0],0),0),[1,1,1,9,1])[:,:,0]
    u1 = simple_conv(fin,tf.expand_dims(tf.expand_dims(v[:,1],0),0),[1,1,1,9,1])[:,:,0]
    u0 /= rho
    u1 /= rho
    return tf.stack([u0,u1],axis=2)

def equilibrium(rho,u) :
    usqr = (3/2 * (u[:,:,0]**2 + u[:,:,1]**2))
    eqs = []
    for i in range(9) :
        cu = 3*(v[i,0]*u[:,:,0] + v[i,1]*u[:,:,1])
        eqs.append(rho*t[i]*(1+cu+0.5*cu**2-usqr))
    return tf.stack(eqs,axis=2)

def inObstacle(x,y) :
    return tf.dtypes.cast((x-cx)**2 + (y-cy)**2 < r**2,tf.dtypes.float32)

def getImage(u) :
    usq =  u**2
    return (2**3)*tf.math.sqrt(simple_conv(usq,tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([tf.dtypes.cast(1.0,tf.float32),tf.dtypes.cast(1.0,tf.float32)]),0),0),[1,1,1,2,1]))

obstacle = np.fromfunction(inObstacle,(nx,ny))

def iniVel(x,y,d) :
    return (1-d)*uLB*(1 + 1e-4*np.sin(y/ly*2*np.pi))

def outflowFin(fin) :
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

def obstacleConv(f) :
    ret = []
    for i in range(9) :
        ret.append(f[:,:,8-i])
    return tf.stack(ret,axis=2)

def obstacleRef(fin, fout, obstacle) :
    a = fout*(1-tf.stack([obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle],axis=2))
    return a + obstacleConv(tf.math.multiply(fin,tf.stack([obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle,obstacle],axis=2)))

def rho0(fin,u) :
    r = tf.zeros((ny))
    for i in range(3) :
        r += ((fin[0,:,col2[i]]) + 2*(fin[0,:,col3[i]])) / (1-u[0,:,0])
    return r

def collideOut(fin,eq) :
    return fin-relax*(fin-eq)

def streamRoller(fout) :
    ret = []
    for i in range(9) :
        ret.append(tf.roll(tf.roll(fout[:,:,i], tf.dtypes.cast(v[i,0],tf.dtypes.int32), axis=0),tf.dtypes.cast(v[i,1],tf.dtypes.int32), axis=1))
    return tf.stack(ret,axis=2)

vel = tf.cast(tf.convert_to_tensor(np.fromfunction(iniVel,(nx,ny,2))),tf.float32)

fininit = tf.zeros((nx,ny,9), dtype=tf.float32)
fininit = equilibrium(1,vel)

fin = tf.Variable(fininit)
fout = tf.Variable(tf.zeros((nx,ny,9), dtype=tf.float32))


u = tf.Variable(tf.zeros((nx,ny,2), dtype=tf.float32))
rho = tf.Variable(tf.zeros((nx,ny), dtype=tf.float32))
eq = tf.Variable(tf.zeros((nx,ny,9), dtype=tf.float32))


def inFin0(fin,eq) :
    return eq[0,:,0] + fin[0,:,8] - eq[0,:,8]
def inFin1(fin,eq) :
    return eq[0,:,1] + fin[0,:,7] - eq[0,:,7]
def inFin2(fin,eq) :
    return eq[0,:,2] + fin[0,:,6] - eq[0,:,6]

outFin = outflowFin(fin)
rhoNew = macroDense(fin)
uNew = macroU(fin,rho)
rho0New = rho0(fin,u)
eqNew = equilibrium(rho,u)
collOut = collideOut(fin,eq)
obstOut = obstacleRef(fin,fout,obstacle)
strOut = streamRoller(fout)



step01 = tf.compat.v1.assign(fin,outflowFin(fin))

with tf.compat.v1.get_default_graph().control_dependencies([step01]):
    step02 = tf.compat.v1.assign(rho,macroDense(fin))

    with tf.compat.v1.get_default_graph().control_dependencies([step02]):
        step03 = tf.compat.v1.assign(u,macroU(fin,rho))

        with tf.compat.v1.get_default_graph().control_dependencies([step03]):
            step04 = tf.compat.v1.assign(u[0,:,:],vel[0,:,:])

            with tf.compat.v1.get_default_graph().control_dependencies([step04]):
                step05 = tf.compat.v1.assign(rho[0,:],rho0(fin,u))
                        
                with tf.compat.v1.get_default_graph().control_dependencies([step05]):
                    step06 = tf.compat.v1.assign(eq,equilibrium(rho,u))

                    with tf.compat.v1.get_default_graph().control_dependencies([step06]):
                        step07 = tf.compat.v1.assign(fin[0,:,0],inFin0(fin,eq))
                        step08 = tf.compat.v1.assign(fin[0,:,1],inFin1(fin,eq))
                        step09 = tf.compat.v1.assign(fin[0,:,2],inFin2(fin,eq))

                        with tf.compat.v1.get_default_graph().control_dependencies([step07,step08,step09]):
                            step10 = tf.compat.v1.assign(fout,collideOut(fin,eq))

                            with tf.compat.v1.get_default_graph().control_dependencies([step10]):
                                step11 = tf.compat.v1.assign(fout,obstacleRef(fin,fout,obstacle))

                                with tf.compat.v1.get_default_graph().control_dependencies([step11]):
                                    step12 = tf.compat.v1.assign(fin,streamRoller(fout))

                                    with tf.compat.v1.get_default_graph().control_dependencies([step12]):
                                        step13 = tf.compat.v1.assign(fin,fin)

combo = tf.group(step13,)


a = tf.group(
    tf.compat.v1.assign(fin,outFin),
)

b = tf.group(
    tf.compat.v1.assign(rho,rhoNew),
)

c = tf.group(
    tf.compat.v1.assign(u,uNew),
)

d = tf.group(
    tf.compat.v1.assign(u[0,:,:],vel[0,:,:]),
)

e = tf.group(
    tf.compat.v1.assign(rho[0,:],rho0New),
)

ff = tf.group(
    tf.compat.v1.assign(eq,eqNew),
)

g = tf.group(
    tf.compat.v1.assign(fin[0,:,0],inFin0(fin,eq)),
    tf.compat.v1.assign(fin[0,:,1],inFin1(fin,eq)),
    tf.compat.v1.assign(fin[0,:,2],inFin2(fin,eq))
)

j = tf.group(
    tf.compat.v1.assign(fout,collOut),

)

k = tf.group(
    tf.compat.v1.assign(fout,obstOut),
)

l = tf.group(
    tf.compat.v1.assign(fin,strOut),
)

tf.compat.v1.global_variables_initializer().run()

for time in range(20000):
    # a.run()
    # b.run()
    # c.run()
    # d.run()
    # e.run()
    # ff.run()
    # g.run()
    # j.run()
    # k.run()
    # l.run()
    sess.run(combo)
    if (time%100 == 0) :
        img = tf.image.encode_png(tf.image.convert_image_dtype(getImage(u),tf.dtypes.uint16))
        f = open("output/vel.{0:04d}.png".format(time//100), "wb+")
        f.write(img.eval())
        f.close()
        