# Creating Lattice Boltzmann Solvers in Python and Tensorflow

This project includes 2D Lattice Boltzmann fluid solvers (https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) in both Tensorflow and only Numpy.

I created these solvers in order to assess the potential improvements that Tensorflow can bring to a method like this. My analysis is available at TODO

## How to Run

The solvers can be run from a terminal.  Before running the solvers, create a directory called `output`.  The output will be a series of .png files in this folder.

The solvers run by default run for 20000 timesteps, and create an image every 100 timesteps, but this can be changed in the constants section of each solver.

To create your own solver, copy the file Experiments/Numpy/D2Q9_NP_Template.py or Experiments/TF/D2Q9_TF_Template.py. The solvers can be edited to have inflow and outflow from any face, and the structure of the map can be read in from a .png file. The lines that must be changed are indicated by a triple hash \#\#\#.


### Prerequisites

The following packages are needed to run the solvers.

For the Numpy solvers:

* numpy

* matplotlib

* opencv

For the Tensorflow solvers:

* numpy

* tensorflow

* opencv


## Sample simulations

There are 6 sample simulations given for both Numpy and Tensorflow.

These are:

* Flow around a cylinder with low viscosity

* Flow around a cylinder with high viscosity

* Flow around an airfoil

* Flow through a narrowing pipe

* Flow through a bending pipe

* Flow in a lid driven cavity


### Animations

The sample simulations produced these animations: TODO

## Acknowledgements

* This was a project conducted in an internship in the Research Software Engineering group at the University of Cambridge, under the supervision of Jeffrey Salmond (@js947)
