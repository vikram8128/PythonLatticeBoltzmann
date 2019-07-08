from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm



#Flow parameters
iterations = 200000
Re = 10.0 #Reynolds number
nx = 420
ny = 180
nz = 180
ly = ny-1 # domain height
lz = nz-1 # domain depth
cx, cy, cz, r = nx//4, ny//2, nz//2, ny//9 # location of obstacle
uLB = 0.04 # inflow velocity
nulb = (uLB*r/Re) # viscosity
relax = 1/(3*nulb+0.5) # relaxation parameter


#Lattice constants
qDim = 1 # number of rings in lattice direction grid

def genV() : # Generator for lattice directions
    v = []
    for x in range(qDim,-qDim-1,-1) :
        for y in range(qDim,-qDim-1,-1) :
            for z in range(qDim,-qDim-1,-1) :
                v.append([x,y,z])
    return v

v = array(genV())

print(v)

#TODO generalise weights


t = [1/216,1/54,1/216, 1/54,2/27,1/54, 1/216,1/54,1/216,
    1/54,2/27,1/54, 2/27,8/27,2/27, 1/54,2/27,1/54,
    1/216,1/54,1/216, 1/54,2/27,1/54, 1/216,1/54,1/216]

faceIn = array([0,1,2,3,4,5,6,7,8])
faceMid = array([9,10,11,12,13,14,15,16,17])
faceOut = array([18,19,20,21,22,23,24,25,26])

def macroDense(fin) :
    rho = sum(fin,axis=0)
    u = zeros((3,nx,ny,nz))
    for i in range(27) :
        u[0,:,:,:] += v[i,0]*fin[i,:,:,:]
        u[1,:,:,:] += v[i,1]*fin[i,:,:,:]
        u[2,:,:,:] += v[i,2]*fin[i,:,:,:]
    u /= rho
    return rho, u

def equilibrium(rho,u) :
    usqr = 3/2 * (u[0]**2 + u[1]**2 + u[2]**2)
    eq = zeros((27,nx,ny,nz))
    for i in range(9) :
        cu = 3*(v[i,0]*u[0,:,:,:] + v[i,1]*u[1,:,:,:] + v[i,2]*u[2,:,:,:])
        eq[i,:,:,:] = rho*t[i]*(1+cu+0.5*cu**2-usqr)
    return eq

# Create Obstacle
def inObstacle(x,y,z) :
    return (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < r**2

obstacle = fromfunction(inObstacle,(nx,ny,nz))

#Initial velocity (slight disturbance to give instability)

def iniVel(d,x,y,z) :
    return (1-d)*uLB*(1 + 1e-4*sin(y/ly*2*pi)) +4

vel = fromfunction(iniVel,(3,nx,ny,nz))

fin = equilibrium(1,vel)

for time in range(iterations) :
    fin[faceOut,-1,:,:] = fin[faceOut,-2,:,:] # outflow

    rho,u = macroDense(fin)

    #inflow
    u[:,0,:,:] = vel[:,0,:,:]
    rho[0,:,:] = ((sum(fin[faceMid,0,:,:],axis=0)) + 2*(sum(fin[faceOut,0,:,:],axis=0))) / (1-u[0,0,:,:]) + 1

    #equlibrium
    eq = equilibrium(rho,u)
    fin[faceIn,0,:,:] = eq[faceIn,0,:,:] + fin[faceOut[::-1],0,:,:] - eq[faceOut[::-1],0,:,:]

    #collision
    fout = fin-relax*(fin-eq) 

    #obstacle
    for i in range(9):
        fout[i,obstacle] = fin[26-i,obstacle]

    #streaming
    for i in range(9):
        fin[i,:,:,:] = roll(roll(roll(fout[i,:,:,:], v[i,0], axis=0), v[i,1], axis=1), v[i,2], axis=2)

    if(time%1==0) :
        plt.clf()
        plt.imshow(sqrt(u[0,:,:,90]**2+u[1,:,:,90]**2+u[2,:,:,90]**2).transpose(),cmap=cm.Blues)
        plt.savefig("vel.{0:04d}.png".format(time//1))