# -*- coding: utf-8 -*-
"""modbinary_functs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hZreriutD3tmqJvucJIV11whE4lfmwTb
"""

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
from typing import Optional, List, Union, Tuple
from network_helpers import (
    tprod, orthogonalize, expand_dims, matricize)

"""
modbinary_functs

- define_ham
- initialize
- define_networks
- lift_hamiltonian
- lower_density
- optimize_w
- optimize_v
- optimize_u
- modbinary_optimize
"""

def define_ham(blocksize):
  """ 
  Define Hamiltonian (quantum critical Ising), perform preliminary blocking 
  of several sites into an effective site.
  """

  # define Pauli matrices
  sX = np.array([[0, 1], [1, 0]], dtype=float)
  sZ = np.array([[1, 0], [0, -1]], dtype=float)

  # define Ising local Hamiltonian
  ham_orig = (tprod(sX, sX) - 0.5*tprod(sZ, np.eye(2)) - 
              0.5*tprod(np.eye(2), sZ))

  # shift Hamiltonian to ensure negative defined
  en_shift = max(LA.eigh(ham_orig)[0])
  ham_loc = ham_orig - en_shift*np.eye(4)

  # define block Hamiltonians 
  d0 = 2 # initial local dim
  d1 = d0**blocksize # local dim after blocking

  if blocksize==2:
    ham_block = (0.5*tprod(ham_loc, np.eye(d0**2)) + 
                 1.0*tprod(np.eye(d0**1), ham_loc, np.eye(d0**1)) +
                 0.5*tprod(np.eye(d0**2), ham_loc)
                 ).reshape(d0*np.ones(8, dtype=int))
    hamAB_init = ham_block.transpose(0,1,4,3,5,6,8,7
                                    ).reshape(d1, d1, d1, d1)
    hamBA_init = ham_block.transpose(1,0,3,4,6,5,7,8
                                    ).reshape(d1, d1, d1, d1)
  elif blocksize==3:
    ham_block = (1.0*tprod(np.eye(d0**1), ham_loc, np.eye(d0**3)) + 
                 1.0*tprod(np.eye(d0**2), ham_loc, np.eye(d0**2)) +
                 1.0*tprod(np.eye(d0**3), ham_loc, np.eye(d0**1))
                 ).reshape(d0*np.ones(12, dtype=int))
    hamAB_init = ham_block.transpose(0,1,2,5,4,3,6,7,8,11,10,9
                                    ).reshape(d1, d1, d1, d1)
    hamBA_init = ham_block.transpose(2,1,0,3,4,5,8,7,6,9,10,11
                                    ).reshape(d1, d1, d1, d1)
  elif blocksize==4:
    ham_block = (0.5*tprod(np.eye(d0**1), ham_loc, np.eye(d0**5)) + 
                1.0*tprod(np.eye(d0**2), ham_loc, np.eye(d0**4)) + 
                1.0*tprod(np.eye(d0**3), ham_loc, np.eye(d0**3)) +
                1.0*tprod(np.eye(d0**4), ham_loc, np.eye(d0**2)) +
                0.5*tprod(np.eye(d0**5), ham_loc, np.eye(d0**1))
                ).reshape(d0*np.ones(16, dtype=int))
    hamAB_init = ham_block.transpose(0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12
                                    ).reshape(d1, d1, d1, d1)
    hamBA_init = ham_block.transpose(3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15
                                    ).reshape(d1, d1, d1, d1)
  
  return hamAB_init, hamBA_init, en_shift

def initialize(chi, chimid, hamAB_init, hamBA_init, layers):
  """ Initialize the MERA tensors """

  # Initialize the MERA tensors
  d1 = hamAB_init.shape[0]
  iso_temp = orthogonalize(np.random.rand(d1, min(chimid, d1)))
  uC = [tprod(iso_temp, iso_temp, do_matricize=False)]
  wC = [orthogonalize(np.random.rand(d1, uC[0].shape[2], chi), partition=2)]
  vC = [orthogonalize(np.random.rand(d1, uC[0].shape[2], chi), partition=2)]
  for k in range(layers-1):
    iso_temp = orthogonalize(np.random.rand(chi, chimid))
    uC.append(tprod(iso_temp, iso_temp, do_matricize=False))
    wC.append(orthogonalize(np.random.rand(chi, chimid, chi), partition=2))
    vC.append(orthogonalize(np.random.rand(chi, chimid, chi), partition=2))
  
  # initialize density matrices and effective Hamiltonians
  rhoAB = [0]
  rhoBA = [0]
  hamAB = [hamAB_init]
  hamBA = [hamBA_init]
  for k in range(layers):
    rhoAB.append(np.eye(chi**2).reshape(chi, chi, chi, chi))
    rhoBA.append(np.eye(chi**2).reshape(chi, chi, chi, chi))
    hamAB.append(np.zeros((chi, chi, chi, chi)))
    hamBA.append(np.zeros((chi, chi, chi, chi)))
  
  return hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA

def define_networks(hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA):
  """ Define and plot all principle networks """

  # Define the `M` principle network
  connects_M = [[3,5,9], [1,5,7], [1,2,3,4], [4,6,10], [2,6,8], [7,8,9,10]]
  tensors_M = [vC, vC, hamBA, wC, wC, rhoAB]
  order_M = ncon_solver(tensors_M, connects_M)[0]
  dims_M = [tensor.shape for tensor in tensors_M]
  names_M = ['v', 'v', 'hBA', 'w', 'w', 'rhoAB']
  coords_M = [(-0.5,1),(-0.5,-1), (-0.3,-0.2,0.3,0.2),(0.5,1),(0.5,-1),(0.2)]
  colors_M = [0,0,1,2,2,3]

  # Define the `L` principle network
  connects_L = [[3,6,13], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4], 
                [10,7,14], [10,9,12], [11,12,13,14]]
  tensors_L = [wC, wC, uC, uC, hamAB, vC, vC, rhoBA]
  order_L = ncon_solver(tensors_L, connects_L)[0]
  dims_L = [tensor.shape for tensor in tensors_L]
  names_L = ['w', 'w', 'u', 'u', 'hAB', 'v', 'v', 'rhoBA']
  coords_L = [(-0.5, 1.5), (-0.5, -1.5), (-0.3,0.5,0.3,0.9), (-0.3,-0.5,0.3,-0.9), 
              (-0.6,-0.2,-0.1,0.2), (0.5, 1.5), (0.5, -1.5), (0.2)]
  colors_L = [2,2,4,4,1,0,0,3]

  # Define the `C` principle network
  connects_C = [[5,6,13], [5,9,11], [3,4,6,8], [1,2,9,10], [1,2,3,4], [7,8,14],
                [7,10,12], [11,12,13,14]]
  tensors_C = [wC, wC, uC, uC, hamBA, vC, vC, rhoBA]
  order_C = ncon_solver(tensors_C, connects_C)[0]
  dims_C = [tensor.shape for tensor in tensors_C]
  names_C = ['w', 'w', 'u', 'u', 'hBA', 'v', 'v', 'rhoBA']
  coords_C = [(-0.5, 1.5), (-0.5, -1.5), (-0.3,0.5,0.3,0.9), (-0.3,-0.5,0.3,-0.9), 
              (-0.3,-0.2,0.3,0.2), (0.5, 1.5), (0.5, -1.5), (0.2)]
  colors_C = [2,2,4,4,1,0,0,3]

  # Define the `R` principle network
  connects_R = [[10,6,13], [10,8,11], [5,3,6,7], [5,1,8,9], [1,2,3,4], [4,7,14],
                [2,9,12], [11,12,13,14]]
  tensors_R = [wC, wC, uC, uC, hamAB, vC, vC, rhoBA]
  order_R = ncon_solver(tensors_R, connects_R)[0]
  dims_R = [tensor.shape for tensor in tensors_R]
  names_R = ['w', 'w', 'u', 'u', 'hAB', 'v', 'v', 'rhoBA']
  coords_R = [(-0.5, 1.5), (-0.5, -1.5), (-0.3,0.5,0.3,0.9), (-0.3,-0.5,0.3,-0.9), 
              (0.6,-0.2,0.1,0.2), (0.5, 1.5), (0.5, -1.5), (0.2)]
  colors_R = [2,2,4,4,1,0,0,3]

  # Plot all principle networks
  fig = plt.figure(figsize=(24,24))
  figM = draw_network(connects_M, order=order_M, dims=dims_M, coords=coords_M, 
                      names=names_M, colors=colors_M, title='M-diagrams', 
                      draw_labels=False, show_costs=True, legend_extend=2.5, 
                      fig=fig, subplot=141, env_pad=(-0.4,-0.4))
  figL = draw_network(connects_L, order=order_L, dims=dims_L, coords=coords_L, 
                      names=names_L, colors=colors_L, title='L-diagrams', 
                      draw_labels=False, show_costs=True, legend_extend=2.5, 
                      fig=fig, subplot=142, env_pad=(-0.4,-0.4))
  figC = draw_network(connects_C, order=order_C, dims=dims_C, coords=coords_C, 
                      names=names_C, colors=colors_C, title='C-diagrams', 
                      draw_labels=False, show_costs=True, legend_extend=2.5, 
                      fig=fig, subplot=143, env_pad=(-0.4,-0.4))
  figR = draw_network(connects_R, order=order_R, dims=dims_R, coords=coords_R, 
                      names=names_R, colors=colors_R, title='R-diagrams', 
                      draw_labels=False, show_costs=True, legend_extend=2.5, 
                      fig=fig, subplot=144, env_pad=(-0.4,-0.4))

  # Store `connects` and `order` in a dict for later use
  network_dict = {'connects_M': connects_M, 'order_M': order_M,
                  'connects_L': connects_L, 'order_L': order_L,
                  'connects_C': connects_C, 'order_C': order_C,
                  'connects_R': connects_R, 'order_R': order_R,}

  return network_dict

def lift_hamiltonian(hamAB, hamBA, w, v, u, rhoAB, rhoBA, network_dict, 
                     ref_sym=False):
  """ Lift the Hamiltonian through one MERA layer """

  hamAB_lift = xcon([v, v, hamBA, w, w, rhoAB], 
                    network_dict['connects_M'], 
                    order=network_dict['order_M'], which_envs=5)

  hamBA_temp0 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                     network_dict['connects_L'], 
                     order=network_dict['order_L'], which_envs=7)

  hamBA_temp1 = xcon([w, w, u, u, hamBA, v, v, rhoBA], 
                     network_dict['connects_C'], 
                     order=network_dict['order_C'], which_envs=7)

  if ref_sym is True:
    hamBA_temp2 = hamBA_temp0.transpose(1,0,3,2)
  else:
    hamBA_temp2 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                      network_dict['connects_R'], 
                      order=network_dict['order_R'], which_envs=7)
  
  hamBA_lift = hamBA_temp0 + hamBA_temp1 + hamBA_temp2

  return hamAB_lift, hamBA_lift

def lower_density(hamAB, hamBA, w, v, u, rhoAB, rhoBA, network_dict, 
                  ref_sym=False):
  """ Lower the density matrix through one MERA layer """

  rhoBA_temp0 = xcon([v, v, hamBA, w, w, rhoAB], 
                     network_dict['connects_M'], 
                     order=network_dict['order_M'], which_envs=2)

  rhoAB_temp0 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                     network_dict['connects_L'], 
                     order=network_dict['order_L'], which_envs=4)

  rhoBA_temp1 = xcon([w, w, u, u, hamBA, v, v, rhoBA], 
                     network_dict['connects_C'], 
                     order=network_dict['order_C'], which_envs=4)
  
  if ref_sym is True:
    rhoAB_temp1 = rhoAB_temp0.transpose(1,0,3,2)
  else:
    rhoAB_temp1 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                      network_dict['connects_R'], 
                      order=network_dict['order_R'], which_envs=4)
  
  rhoAB_lower = 0.5*(rhoAB_temp0 + rhoAB_temp1)
  rhoBA_lower = 0.5*(rhoBA_temp0 + rhoBA_temp1)

  return rhoAB_lower, rhoBA_lower

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
  
  w_out = orthogonalize(w_env0 + w_env1 + w_env2 + w_env3, partition=2)

  return w_out

def optimize_v(hamAB, hamBA, w, v, u, rhoAB, rhoBA, network_dict, 
               ref_sym=False):
  """ Optimise the `v` isometry """

  v_env0 = xcon([v, v, hamBA, w, w, rhoAB], network_dict['connects_M'], 
                order=network_dict['order_M'], which_envs=0)
  
  if ref_sym is True:
    v_env1, v_env3 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                          network_dict['connects_L'], 
                          order=network_dict['order_L'], 
                          which_envs=[0,5])
  else:
    v_env1 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                  network_dict['connects_L'], order=network_dict['order_L'], 
                  which_envs=5)
    
    v_env3 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                  network_dict['connects_R'], order=network_dict['order_R'], 
                  which_envs=5)
  
  v_env2 = xcon([w, w, u, u, hamBA, v, v, rhoBA], 
                network_dict['connects_C'], order=network_dict['order_C'], 
                which_envs=5)
  
  v_out = orthogonalize(v_env0 + v_env1 + v_env2 + v_env3, partition=2)

  return v_out

def optimize_u(hamAB, hamBA, w, v, u, rhoAB, rhoBA, network_dict, 
               ref_sym=False):
  """ Optimise the `u` disentangler """
  
  u_env0 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                network_dict['connects_L'], order=network_dict['order_L'], 
                which_envs=2)
  
  u_env1 = xcon([w, w, u, u, hamBA, v, v, rhoBA], 
                network_dict['connects_C'], order=network_dict['order_C'], 
                which_envs=2)
  
  if ref_sym is True:
    u_env2 = u_env0.transpose(1,0,3,2)
  else:
    u_env2 = xcon([w, w, u, u, hamAB, v, v, rhoBA], 
                  network_dict['connects_R'], order=network_dict['order_R'], 
                  which_envs=2)
  
  utot = u_env0 + u_env1 + u_env2
  if ref_sym is True:
    utot = utot + utot.transpose(1,0,3,2)
  
  u_out = orthogonalize(utot, partition=2)

  return u_out

def modbinary_optimize(hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, network_dict, 
                       chi=None, chimid=None, layers=None, scale_iters=3, 
                       mtype='finite', display_step=10, en_shift=0, en_exact=0,
                       blocksize=1, iterations=100):

  # add extra layers if necessary
  if layers is not None:
    for k in range(layers - len(wC)):
      wC.append(wC[-1])
      vC.append(vC[-1])
      uC.append(uC[-1])
      hamAB.append(hamAB[-1])
      hamBA.append(hamBA[-1])
      rhoAB.append(rhoAB[-1])
      rhoBA.append(rhoBA[-1])
  else:
    layers = len(wC)

  # expand tensors to new dims if necessary
  if (chi is not None) and (chimid is not None):
    for z in range(layers):
      hamAB[z+1] = expand_dims(hamAB[z+1], (chi,chi,chi,chi))
      hamBA[z+1] = expand_dims(hamBA[z+1], (chi,chi,chi,chi))
      rhoAB[z+1] = expand_dims(rhoAB[z+1], (chi,chi,chi,chi))
      rhoBA[z+1] = expand_dims(rhoBA[z+1], (chi,chi,chi,chi))
      if z == 0:
        chitemp = min(2**blocksize, chimid)
        uC[z] = expand_dims(uC[z], (2**blocksize, 2**blocksize, chitemp, chitemp))
        wC[z] = expand_dims(wC[z], (2**blocksize, chitemp, chi))
        vC[z] = expand_dims(vC[z], (2**blocksize, chitemp, chi))
      else:
        uC[z] = expand_dims(uC[z], (chi, chi, chimid, chimid))
        wC[z] = expand_dims(wC[z], (chi, chimid, chi))
        vC[z] = expand_dims(vC[z], (chi, chimid, chi))

  # start variational iterations
  num_sites = 2*blocksize*(2**layers)
  for iter in range(iterations):

    # sweep over all layers
    for z in range(layers):

      # optimize isometries
      if iter > 2:
        wC[z] = optimize_w(hamAB[z], hamBA[z], wC[z], vC[z], uC[z], rhoAB[z+1], 
                          rhoBA[z+1], network_dict, ref_sym=ref_sym)      
        if ref_sym is True:
          vC[z] = wC[z]
        else:
          vC[z] = optimize_v(hamAB[z], hamBA[z], wC[z], vC[z], uC[z], rhoAB[z+1], 
                            rhoBA[z+1], network_dict, ref_sym=ref_sym)
      
      # optimize disentanglers
      if iter > 10:
        uC[z] = optimize_u(hamAB[z], hamBA[z], wC[z], vC[z], uC[z], rhoAB[z+1], 
                          rhoBA[z+1], network_dict, ref_sym=ref_sym)
      
      # lift Hamiltonian
      hamAB[z+1], hamBA[z+1] = lift_hamiltonian(hamAB[z], hamBA[z], wC[z], vC[z], 
                                                uC[z], rhoAB[z+1], rhoBA[z+1], 
                                                network_dict, ref_sym=ref_sym)

    # find top-layer density matrices
    if mtype == 'scale':
      # find scale-invariant density matrix (power method)
      for k in range(scale_iters):
        rhoAB_temp, rhoBA_temp = lower_density(hamAB[layers-1], hamBA[layers-1], 
                                              wC[layers-1], vC[layers-1], 
                                              uC[layers-1], rhoAB[layers], 
                                              rhoBA[layers], network_dict, 
                                              ref_sym=ref_sym)
        # ensure Hermitcity
        rhoAB_temp = rhoAB_temp + rhoAB_temp.transpose(2,3,0,1)
        rhoBA_temp = rhoBA_temp + rhoBA_temp.transpose(2,3,0,1)

        if ref_sym:
          # ensure reflection symmetric
          rhoAB_temp = rhoAB_temp + rhoAB_temp.transpose(1,0,3,2)
          rhoBA_temp = rhoBA_temp + rhoBA_temp.transpose(1,0,3,2)

      rhoAB[layers] = rhoAB_temp / np.trace(matricize(rhoAB_temp))
      rhoBA[layers] = rhoBA_temp / np.trace(matricize(rhoBA_temp))

    elif mtype == 'finite':
      # diagonalize finite Hamiltonian for ground state
      H = matricize(hamAB[layers] + hamBA[layers].transpose(1,0,3,2))
      energy_top, psi = eigsh(H, k=1, which='SA')

      chitemp = hamAB[layers].shape[0]
      rhoAB[layers] = np.outer(psi,psi).reshape(chitemp,chitemp,chitemp,chitemp)
      rhoBA[layers] = rhoAB[layers].transpose(1,0,3,2)

    # lower density matrix through all layers
    for z in reversed(range(layers)):
      rhoAB[z], rhoBA[z] = lower_density(hamAB[z], hamBA[z], wC[z], vC[z], uC[z], 
                                        rhoAB[z+1], rhoBA[z+1], network_dict,
                                        ref_sym=ref_sym)
    
    # evaluate energy
    if np.remainder(iter,display_step) == 1:
      energy0 = (np.trace(matricize(rhoAB[0]) @ matricize(hamAB[0])) +
                np.trace(matricize(rhoBA[0]) @ matricize(hamBA[0])))
      energy = 0.5*(energy0 / blocksize) + en_shift
      log_err = -np.log10(energy - en_exact)
      print('Iter {iter} of {iterations}, Energy: {energy:.10f}, Log10-Err: {log_err:0.3f}'
          .format(iter=iter, iterations=iterations, energy=energy, 
                  log_err=log_err))
  
  return hamAB, hamBA, wC, vC, uC, rhoAB, rhoBA, energy