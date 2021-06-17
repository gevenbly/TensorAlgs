# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 02:04:27 2021

@author: gevenbly3
"""

from timeit import default_timer as timer
from network_contract import ncon
from timeit import default_timer as timer
from network_contract import xcon, remove_tensor
from modbinary_functs import define_ham, initialize, define_networks

chi = 16 # MERA bond dimension between layers 
chimid = 12 # MERA bond dimension within middle of layer

ref_sym = True  # impose reflection symmetry
layers = 2  # total number of unique MERA layers (minimum: layers=2)
blocksize = 2 # size of blocks in prelim coarse-graining

# define the Hamiltonian
hamAB_init, hamBA_init, en_shift = define_ham(blocksize)
# initialize the tensors
hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA = initialize(
    chi, chimid, hamAB_init, hamBA_init, layers)
network_dict = define_networks(hamAB[1], hamBA[1], wC[1], vC[1], uC[1], 
                               rhoAB[2], rhoBA[2])


def optimize_all(hamAB, hamBA, w, v, u, rhoAB, rhoBA, network_dict, 
               ref_sym=False):
  
  
  w_env, u_env, ham_env, v_env, rho_env = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                        network_dict['connects_L'], 
                        order=network_dict['order_L'], 
                        which_envs=[0,2,4,5,7])

  return w_env, u_env, ham_env, v_env, rho_env


def optimize_w(hamAB, hamBA, w, v, u, rhoAB, rhoBA, network_dict, 
               ref_sym=False):
  """ Optimise the `w` isometry """

  w_env0 = xcon([v, v, hamBA, w, w, rhoAB], network_dict['connects_M'], 
                order=network_dict['order_M'], which_envs=3)
  
  if ref_sym is True:
    w_env1, w_env3 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                          network_dict['connects_L'], 
                          order=network_dict['order_L'], 
                          which_envs=[0,5])
  else:
    w_env1 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                  network_dict['connects_L'], order=network_dict['order_L'], 
                  which_envs=0)
    
    w_env3 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                network_dict['connects_R'], order=network_dict['order_R'], 
                which_envs=0)
  
  w_env2 = xcon([w, w, u, u, hamBA, v, v, rhoBA], 
                network_dict['connects_C'], order=network_dict['order_C'], 
                which_envs=0)
  
  w_out = w_env0 + w_env1 + w_env2 + w_env3

  return w_out

for k in range(100):
  st = timer()
  z = 1
  w_env, u_env, ham_env, v_env, rho_env = optimize_all(hamAB[z+1], hamBA[z+1], wC[z], vC[z], uC[z], rhoAB[z+1], rhoBA[z+1], network_dict)
  # optimize_w(hamAB[z+1], hamBA[z+1], wC[z], vC[z], uC[z], rhoAB[z+1], rhoBA[z+1], network_dict)
  # connects_new, order_new, open_ord = remove_tensor(network_dict['connects_L'], 
  #                                                   which_env=0, order=network_dict['order_L'])
  # wEnv, order, cost = xcon([wC[z], wC[z], uC[z], uC[z], hamAB[z], vC[z], vC[z], rhoBA[z+1]], 
  #                   network_dict['connects_L'], order=network_dict['order_L'], 
  #                   which_envs=0, return_info=True, standardize_inputs=False,
  #                   perform_check=False)
  ft = timer() - st
  print(ft)