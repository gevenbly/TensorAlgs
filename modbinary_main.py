
# Import necessary modules
import numpy as np
from modbinary_functs import (
    define_ham, define_networks, modbinary_optimize, initialize)

""" Specify options """
chi = 6 # MERA bond dimension between layers 
chimid = 4 # MERA bond dimension within middle of layer

ref_sym = True  # impose reflection symmetry
layers = 2  # total number of unique MERA layers (minimum: layers=2)
blocksize = 2 # size of blocks in prelim coarse-graining

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
iterations = 2000

hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy = modbinary_optimize(
    hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, mtype='scale', 
    display_step=10, en_shift=en_shift, en_exact=en_exact, blocksize=blocksize,
    iterations=iterations, ref_sym=ref_sym, chi=chi, chimid=chimid, 
    layers=layers)

# perform variational energy minimization
iterations = 1800
chi = 8 
chimid = 6

hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy = modbinary_optimize(
    hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, mtype='scale', 
    display_step=10, en_shift=en_shift, en_exact=en_exact, blocksize=blocksize,
    iterations=iterations, ref_sym=ref_sym, chi=chi, chimid=chimid, 
    layers=layers)

# perform variational energy minimization
iterations = 1400
layers = 3
chi = 12
chimid = 8

hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy = modbinary_optimize(
    hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, mtype='scale', 
    display_step=10, en_shift=en_shift, en_exact=en_exact, blocksize=blocksize,
    iterations=iterations, ref_sym=ref_sym, chi=chi, chimid=chimid, 
    layers=layers)

# perform variational energy minimization
iterations = 1400
layers = 4
chi = 16
chimid = 12

hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy = modbinary_optimize(
    hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, mtype='scale', 
    display_step=10, en_shift=en_shift, en_exact=en_exact, blocksize=blocksize,
    iterations=iterations, ref_sym=ref_sym, chi=chi, chimid=chimid, 
    layers=layers)

