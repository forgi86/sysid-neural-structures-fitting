# Efficient Calibration of Embedded MPC

This repository contains the Python code to reproduce the results of the paper "Efficient Calibration of Embedded MPC" by Marco Forgione and Dario Piga

The following fitting methods for State-Space (SS) and Input-Output (IO) neural dynamical models are developed based on

 * One-step prediction error minimization
 * Open-loop simulation error minimization
 * Multistep simulation error minimization

## Example folders: 
The examples performed are the following:

* `` RLC_example``: nonlinear RLC circuit thoroughly discussed in the paper
* `` CSTR_example``: CSTR system from 
* `` cartpole_example``: cart-pole system

For the RLC example, the main scripts are:

 * ``RLC_generate_id.py``:  generate the identification dataset 
 * ``RLC_generate_val.py``: generate the validation dataset 

For the SS model
 *  ``RLC_SS_fit_1step.py``: SS model, one-step prediction error minimization
 *  ``RLC_SS_fit_simerror.py``: SS model, open-loop simulation error minimization
 *  ``RLC_SS_fit_multistep.py``: SS model, multistep simulation error minimization
 *  ``RLC_SS_eval_sim.py``: SS model evaluate the simulation performance of the identified models, produce relevant plots 
 and model statistics
 
  
Simulations were performed on a Python 3.7 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * sympy
 * numba
 * pytorch
 
These dependencies may be installed through the commands:
```
conda install numpy scipy matplotlib pandas
```
