# Creating Lattice Boltzmann Solvers in Python and Tensorflow

This project includes 2D Lattice Boltzmann fluid solvers (https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) in both Tensorflow and only Numpy.

I created these solvers in order to assess the potential improvements that Tensorflow can bring to a method like this. A short report on this project is in [Analysis.pdf](Analysis/Analysis.pdf)

## How to Run

The solvers can be run from a command line. Before running the solvers, create a directory called `output`. The output will be a series of .png files in this folder.

The solvers run by default run for 20000 timesteps, and create an image every 100 timesteps, but this can be changed in the constants section of each solver.

To create your own solver, copy the file [D2Q9_NP_Template.py](Experiments/Numpy/D2Q9_NP_Template.py) or [D2Q9_TF_Template.py](Experiments/TensorFlowCPU/D2Q9_TF_Template.py). The solvers can be edited to have inflow and outflow from any face, and the structure of the map can be read in from a .png file. The lines that must be changed are indicated by a triple hash `###`.


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

The sample simulations produced these animations:

| Numpy | Tensorflow |
| --- | --- |
| ![](Videos/NP_AirfoilFlow.gif) | ![](Videos/TF_Airfoilflow.gif) |
| ![](Videos/NP_BendFlow.gif) | ![](Videos/TF_BendFlow.gif) |
| ![](Videos/NP_CylinderFlow.gif) | ![](Videos/TF_CylinderLowViscosity.gif) |
| ![](Videos/NP_CylinderFlowHighViscosity.gif) | ![](Videos/TF_CylinderHighViscosity.gif) |
| ![](Videos/NP_PipeFlow.gif) | ![](Videos/TF_PipeFlow.gif) |


## Acknowledgements

* This was a project conducted in an internship in the Research Software Engineering group at the University of Cambridge, under the supervision of Jeffrey Salmond (@js947)
