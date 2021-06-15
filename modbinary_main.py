# -*- coding: utf-8 -*-
"""modbinary_main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BMXOWHbeROR9see_Oh8A2CgLBvguH1-p
"""

# Install glens tensor algorithm library (only for 1st time)
!git clone https://github.com/gevenbly/TensorAlgs

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
import os
os.chdir('/content/TensorAlgs')
from modbinary_functs import (
    define_ham, define_networks, modbinary_optimize, initialize, 
    lift_hamiltonian, lower_density, optimize_w, optimize_v,
    optimize_u, modbinary_optimize)
from network_helpers import (
    tprod, orthogonalize, expand_dims, matricize)

# Alternative to importing individual functions: run an entire notebook

# %run ./network_render.ipynb
# %run ./network_contract.ipynb
# %run ./modbinary_functs.ipynb
# %run ./tensor_helpers.ipynb

""" Specify options """
chi = 6 # MERA bond dimension between layers 
chimid = 4 # MERA bond dimension within middle of layer

ref_sym = True  # impose reflection symmetry
layers = 2  # total number of unique MERA layers (minimum: layers=2)
blocksize = 3 # size of blocks in prelim coarse-graining

""" Setup required before optimization algorithm """ 
# define the Hamiltonian
hamAB_init, hamBA_init, en_shift = define_ham(blocksize)
# initialize the tensors
hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA = initialize(
    chi, chimid, hamAB_init, hamBA_init, layers)
# define, solve and plot the networks
network_dict = define_networks(hamAB[1], hamBA[1], wC[1], vC[1], uC[1], 
                               rhoAB[2], rhoBA[2])

# exact energy for N=inf quantum critical Ising model
en_exact = -4 / np.pi

# perform variational energy minimization
iterations = 500

hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy = modbinary_optimize(
    hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, mtype='scale', 
    display_step=10, en_shift=en_shift, en_exact=en_exact, blocksize=blocksize,
    iterations=iterations)

# exact energy for finite-N quantum critical Ising model
num_sites = 2*blocksize*(2**layers)
en_exact = -2/(num_sites * np.sin(np.pi/(2*num_sites)))

# perform variational energy minimization
iterations = 500

hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy = modbinary_optimize(
    hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, mtype='finite', 
    display_step=10, en_shift=en_shift, en_exact=en_exact, blocksize=blocksize,
    iterations=iterations)

# chi = 24
# chimid = 16
# hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy = modbinary_optimize(
#     hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, mtype='finite', 
#     display_step=10, en_shift=en_shift, en_exact=en_exact, blocksize=blocksize,
#     chi=chi, chimid=chimid)