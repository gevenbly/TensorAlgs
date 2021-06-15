# -*- coding: utf-8 -*-
"""test_network_contract.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19dx8Mv1q3gKAP7gRblwqT8jz6ZAY9zVT
"""

# uncomment code below if running locally

# !git clone https://github.com/gevenbly/TensorAlgs
# import os
# os.chdir('/content/TensorAlgs')

import numpy as np
from typing import Optional, List, Union, Tuple
from network_contract import (
    find_nodes, reorder_nodes, node_to_order, make_canon_connects, 
    make_canon_dims, check_network, compute_costs, remove_tensor, ncon,
    partial_trace, xcon)

"""
Unit tests for 'network_contract'

- find_nodes
- reorder_nodes
- node_to_order
- make_canon_connects
- make_canon_dims
- check_network
- compute_costs
- remove_tensor
- ncon
- partial_trace
- xcon
"""

# unit test: `find_nodes`
connects0 = [[4,5,6], [5,1,2], [6,1,3]]
order0 = [5,6,1]
nodes0 = find_nodes(connects0, order0)[0]
nodes0_ex = np.array([[1,2,4]], dtype=np.uint64)
assert np.array_equal(nodes0, nodes0_ex), (
    'mis-match in nodes: {n0} vs {n1}'.format(n0=nodes0, n1=nodes0_ex))

connects1 = [[3,6,13], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4], [10,7,14],
             [10,9,12], [11,12,13,14]]
order1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
nodes1 = find_nodes(connects1, order1)[0]
nodes1_ex = np.array([[2, 16, 237],
                      [8, 18, 229],
                      [1, 26, 228],
                      [4, 27, 224],
                      [31, 32, 192],
                      [63, 64, 128]], dtype=int)
assert np.array_equal(nodes1, nodes1_ex), (
    'mis-match in nodes: {n0} vs {n1}'.format(n0=nodes1, n1=nodes1_ex))

# unit test: `reorder_nodes`
nodes0 = np.array([[2, 16, 237],
                   [8, 18, 229],
                   [1, 26, 228],
                   [4, 27, 224],
                   [31, 32, 192],
                   [63, 64, 128]], dtype=int)
which_envs0 = 0
node_labs0, needed_conts0 = reorder_nodes(nodes0, which_envs0)
node_labs0_ex = np.array([2,4,8,16,18,26,32,64,128,192,224,228,254], dtype=int)
needed_conts0_ex = np.array([[1,0,0],
                             [1,0,0],
                             [0,0,1],
                             [0,1,0],
                             [0,0,1],
                             [0,0,1]],dtype=bool)
assert np.array_equal(node_labs0, node_labs0_ex), (
    'mis-match in node labels: {n0} vs {n1}'.format(
        n0=node_labs0, n1=node_labs0_ex))
assert np.array_equal(needed_conts0, needed_conts0_ex), (
    'mis-match in needed node contractions: {n0} vs {n1}'.format(
        n0=needed_conts0, n1=needed_conts0_ex))

nodes1 = np.array([[2, 16, 237],
                   [8, 18, 229],
                   [1, 26, 228],
                   [4, 27, 224],
                   [31, 32, 192],
                   [63, 64, 128]], dtype=int)
which_envs1 = [1,3,5]
node_labs1, needed_conts1 = reorder_nodes(nodes1, which_envs1)
node_labs1_ex = np.array([1,2,4,8,16,18,26,27,31,32,64,128,192,223,224,228,229,
                          237,247,253], dtype=int)
needed_conts1_ex = np.array([[1,0,1],
                             [1,1,1],
                             [1,1,0],
                             [1,1,0],
                             [0,1,1],
                             [0,0,1]],dtype=bool)
assert np.array_equal(node_labs1, node_labs1_ex), (
    'mis-match in node labels: {n0} vs {n1}'.format(
        n0=node_labs1, n1=node_labs1_ex))
assert np.array_equal(needed_conts1, needed_conts1_ex), (
    'mis-match in needed node contractions: {n0} vs {n1}'.format(
        n0=needed_conts1, n1=needed_conts1_ex))

# unit test: `node_to_order`
connects = [[3,6,13], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4], [10,7,14],
            [10,9,12], [11,12,13,14]]
nodes = np.array([[2, 16, 237],
                  [8, 18, 229],
                  [1, 26, 228],
                  [4, 27, 224],
                  [31, 32, 192],
                  [63, 64, 128]], dtype=int)
which_env = 0
order = node_to_order(connects, nodes, which_env)
order_ex = [1, 2, 8, 12, 10, 14, 7, 4, 5, 9, 11]
assert order==order_ex, ('mis-match orders: {n0} vs {n1}'.format(
    n0=order, n1=order_ex))

connects = [[3,6,13], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4], [10,7,14],
            [10,9,12], [11,12,13,14]]
nodes = np.array([[2, 16, 237],
                  [8, 18, 229],
                  [1, 26, 228],
                  [4, 27, 224],
                  [31, 32, 192],
                  [63, 64, 128]], dtype=int)
which_env = 3
order = node_to_order(connects, nodes, which_env)
order_ex = [1, 12, 10, 14, 7, 6, 13, 3, 4, 11]
assert order==order_ex, ('mis-match orders: {n0} vs {n1}'.format(
    n0=order, n1=order_ex))

# unit test: `make_canon_connects`
connects = [['alpha','alp','beta'],['beta','gam',2,3],['alp',1,3,4]]
can_connects, fwd_dict, rev_dict = make_canon_connects(connects, one_based=True)[:3]
can_connects_ex = [np.array([-1,1,2]), np.array([2,-2,-4,3]), 
                   np.array([1,-5,3,-3])] 
# test fwd_dict
can_connects0 = []
for connect in connects:
  can_connects0.append(np.array([fwd_dict[ele] for ele in connect]))
# test rev_dict
connects0 = []
for connect in can_connects0:
  connects0.append(np.array([rev_dict[ele] for ele in connect]))

assert np.all([np.array_equal(can_connects[k], can_connects0[k]) for 
               k in range(len(connects))]), (
    'mis-match in canonical connects: {n0} vs {n1}'.format(
        n0=can_connects, n1=can_connects0))

assert np.all([np.array_equal(can_connects[k], can_connects_ex[k]) for 
               k in range(len(connects))]), (
    'mis-match in canonical connects: {n0} vs {n1}'.format(
        n0=can_connects, n1=can_connects_ex))
               
assert np.all([np.array_equal(connects[k], connects0[k]) for 
               k in range(len(connects))]), (
    'mis-match in canonical connects: {n0} vs {n1}'.format(
        n0=connects, n1=connects0))
               
connects = [['alpha','alp','beta'],['beta','gam',2,3],['alp',1,3,4]]
order = [3,'beta','alp','gam']
open_order = [4,1,2,'alpha']
can_connects, fwd_dict, rev_dict = make_canon_connects(
    connects, order=order, open_order=open_order)[:3]
can_connects_ex = [np.array([-3,3,2]), np.array([2,4,-2,1]), 
                   np.array([3,-1,1,-0])] 
# test fwd_dict
can_connects0 = []
for connect in connects:
  can_connects0.append(np.array([fwd_dict[ele] for ele in connect]))
# test rev_dict
connects0 = []
for connect in can_connects0:
  connects0.append(np.array([rev_dict[ele] for ele in connect]))

assert np.all([np.array_equal(can_connects[k], can_connects0[k]) for 
               k in range(len(connects))]), (
    'mis-match in canonical connects: {n0} vs {n1}'.format(
        n0=can_connects, n1=can_connects0))

assert np.all([np.array_equal(can_connects[k], can_connects_ex[k]) for 
               k in range(len(connects))]), (
    'mis-match in canonical connects: {n0} vs {n1}'.format(
        n0=can_connects, n1=can_connects_ex))
               
assert np.all([np.array_equal(connects[k], connects0[k]) for 
               k in range(len(connects))]), (
    'mis-match in canonical connects: {n0} vs {n1}'.format(
        n0=connects, n1=connects0))

# unit test: make_canon_dims
dims = [[1,2,3,'c'],[3,2,'d'],[1,'c','d','x']]
can_dims, fwd_dict, rev_dict = make_canon_dims(dims)

can_dims_ex = [np.array([0, 1, 2, 3]), np.array([2, 1, 4]), np.
               array([0, 3, 4, 5])]
fwd_dict_ex = dict(zip([1,2,3,'c','d','x'], [0,1,2,3,4,5]))
rev_dict_ex = dict(zip([0,1,2,3,4,5], [1,2,3,'c','d','x']))

assert all([np.array_equal(can_dims[k],can_dims_ex[k]) for 
            k in range(len(can_dims))]), (
    'canon dims do not match: {n0} vs {n1}'.format(n0=can_dims, n1=can_dims_ex))

assert fwd_dict == fwd_dict_ex, (
    'dicts do not match: {n0} vs {n1}'.format(n0=fwd_dict, n1=fwd_dict_ex))

assert rev_dict == rev_dict_ex, (
    'dicts do not match: {n0} vs {n1}'.format(n0=rev_dict, n1=rev_dict_ex))

# unit test: `check_network`
connects = [[3,6,13,15], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4,15],
            [10,7,14], [10,9,12], [11,12,13,14]]
order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
dims = [[4,4,4,4], [4,4,4], [4,4,4,4], [4,4,4,4], [4,4,4,4,4],
        [4,4,4], [4,4,4], [4,4,4,4]]
t0 = check_network(connects, dims, order)
assert t0 is True

connects = [['a',6,13,15], [1,8,11], [4,5,6,'c'], [2,5,8,9], [1,2,'a',4,15],
            [10,'c',14], [10,9,'b'], [11,'b',13,14]]
order = [1, 2, 'a', 4, 5, 6, 'c', 8, 9, 10, 11, 'b', 13, 14, 15]
dims = [[4,4,4,4], [4,4,4], [4,4,4,4], [4,4,4,4], [4,4,4,4,4],
        [4,4,4], [4,4,4], [4,4,4,4]]
nml_connects, fwd_dict = make_canon_connects(connects)[:2]
nml_order = [fwd_dict[ele] for ele in order]
t0 = check_network(nml_connects, dims, nml_order)
assert t0 is True

# unit test: `compute_costs`
connects = [[3,6,13,15,16,16], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4,15],
            [10,7,14], [10,9,12], [11,12,13,14]]
dims = [['d',2,2,2,2,2],[2,2,2],[2,2,2,2],['x',2,2,2],[2,2,'d','x',2],
        [2,2,2],[2,2,2],[2,2,2,2]]
bn_costs, pt_costs = compute_costs(connects, dims=dims, return_pt=True)

bn_costs_ex = ['32*(d^1)(x^1)', '64*(d^1)(x^1)', '64*(d^1)(x^1)', '64*(x^1)', 
               64, 64, 16]
pt_costs_ex = ['16*(d^1)']

# mixed symbolic and numeric
assert np.all([bn_costs[k] == bn_costs_ex[k] for k in range(len(bn_costs))]), (
    'incorrect computational costs: {n0} vs {n1}'.format(
        n0=bn_costs, n1=bn_costs_ex))

assert np.all([pt_costs[k] == pt_costs_ex[k] for k in range(len(pt_costs))]), (
    'incorrect computational costs: {n0} vs {n1}'.format(
        n0=pt_costs, n1=pt_costs_ex))

connects = [[3,6,13,15,16,16], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4,15],
            [10,7,14], [10,9,12], [11,12,13,14]]
dims = [['d',2,2,2,2,2],[2,2,2],[2,2,2,2],['x',2,2,2],[2,2,'d','x',2],
        [2,2,2],[2,2,2],[2,2,2,2]]
bn_costs, pt_costs = compute_costs(connects, return_pt=True)

bn_costs_ex = ['(d^7)', '(d^8)', '(d^8)', '(d^7)', '(d^6)', '(d^6)', '(d^4)']
pt_costs_ex = ['(d^5)']

# only symbolic
assert np.all([bn_costs[k] == bn_costs_ex[k] for k in range(len(bn_costs))]), (
    'incorrect computational costs: {n0} vs {n1}'.format(
        n0=bn_costs, n1=bn_costs_ex))

assert np.all([pt_costs[k] == pt_costs_ex[k] for k in range(len(pt_costs))]), (
    'incorrect computational costs: {n0} vs {n1}'.format(
        n0=pt_costs, n1=pt_costs_ex))

connects = [[3,6,13,15,16,16], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,3,4,15],
            [10,7,14], [10,9,12], [11,12,13,14]]
dims = [[20,30,2,50,2,2],[2,2,2],[2,2,30,2],[2,2,2,2],[2,2,20,2,50],
        [2,2,2],[2,2,2],[2,2,2,2]]
bn_costs, pt_costs = compute_costs(connects, dims=dims, return_pt=True)

bn_costs_ex = [32000, 64000, 960000, 1920, 64, 64, 16]
pt_costs_ex = [120000]

# only numeric
assert np.all([bn_costs[k] == bn_costs_ex[k] for k in range(len(bn_costs))]), (
    'incorrect computational costs: {n0} vs {n1}'.format(
        n0=bn_costs, n1=bn_costs_ex))

assert np.all([pt_costs[k] == pt_costs_ex[k] for k in range(len(pt_costs))]), (
    'incorrect computational costs: {n0} vs {n1}'.format(
        n0=pt_costs, n1=pt_costs_ex))

# unit test: `remove_tensor`
connects = [['a',6,13,'b'], [1,8,11], [4,5,6,7], [2,5,8,9], [1,2,'a',4,'b'],
            [10,7,14], [10,9,12], [11,12,13,14,16,16]]
which_env = 2
order = [1, 2, 'a', 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'b', 16]
N = len(connects)

new_connects, new_order, open_ord = remove_tensor(
    connects, which_env, order=order, standardize_outputs=False)
new_connects_ex = [connect for connect in connects]

open_ord_ex = new_connects_ex.pop(which_env)

new_order_ex = [1, 2, 8, 'a', 'b', 12, 10, 14, 9, 11, 13]

orig_cost = compute_costs(connects, order)[:(N-2)]
orig_cost.sort()
new_cost = compute_costs(new_connects, new_order)
new_cost.sort()

assert np.all([new_connects[k] == new_connects_ex[k] for 
               k in range(len(new_connects))]), (
    'incorrect connects: {n0} vs {n1}'.format(
        n0=new_connects, n1=new_connects_ex))

assert open_ord == open_ord_ex, (
    'incorrect open index order: {n0} vs {n1}'.format(
        n0=open_ord, n1=open_ord_ex))

assert new_order == new_order_ex, (
    'incorrect new contraction order: {n0} vs {n1}'.format(
        n0=new_order, n1=new_order_ex))

assert new_cost == orig_cost, (
    'cost should remain the same but does not: {n0} vs {n1}'.format(
        n0=new_cost, n1=orig_cost))

new_connects, new_order, open_ord = remove_tensor(
    connects, which_env, order=order, standardize_outputs=True, one_based=True)

new_connects_ex = [[1, -3, 14, 2], [3, 9, 12], [4, -2, 9, 10], [3, 4, 1, -1, 2], 
                   [11, -4, 15], [11, 10, 13], [12, 13, 14, 15, 16, 16]]
new_order_ex = [3, 4, 9, 1, 2, 13, 11, 15, 10, 12, 14]
open_ord_ex = [-1, -2, -3, -4, -5, -6, -7]

assert np.all([new_connects[k] == new_connects_ex[k] for 
               k in range(len(new_connects))]), (
    'incorrect connects: {n0} vs {n1}'.format(
        n0=new_connects, n1=new_connects_ex))

assert open_ord == open_ord_ex, (
    'incorrect open index order: {n0} vs {n1}'.format(
        n0=open_ord, n1=open_ord_ex))

assert new_order == new_order_ex, (
    'incorrect new contraction order: {n0} vs {n1}'.format(
        n0=new_order, n1=new_order_ex))

# unit test: `ncon`

# closed network with costs
u = np.eye(4).reshape(2,2,2,2)
w = np.eye(4).reshape(2,2,4)
ham = np.random.rand(2,2,2,2,2,2)
rho = np.eye(64).reshape(4,4,4,4,4,4)
tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, 21], [11, 12, 22],
            [13, 14, 23], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, 18], [17, 16, 19], [15, 14, 20], 
            [18, 19, 20, 21, 22, 23]]
order = [4,7,5,6,12,11,3,1,2,17,16,8,10,9,14,13,15,18,19,20,21,22,23]
scalar_out, _, cost = ncon(tensors, connects, order=order, return_info=True)

dims = [tensor.shape for tensor in tensors]
bn_costs, pt_costs = compute_costs(connects, order=order, dims=dims, 
                                   return_pt=True)
cost_ex = sum(bn_costs) + sum(pt_costs)
scalar_ex = ncon([ham], [[1,2,3,1,2,3]]) * 8

assert np.allclose(scalar_out, scalar_ex), (
    'incorrect network product: {n0} vs {n1}'.format(
        n0=scalar_out, n1=scalar_ex))

assert cost == cost_ex, (
    'incorrect contraction cost: {n0} vs {n1}'.format(
        n0=cost, n1=cost_ex))

# open network with final permutation
u = np.eye(4).reshape(2,2,2,2)
w = np.eye(4).reshape(2,2,4)
ham = np.eye(8).reshape(2,2,2,2,2,2)
tensors = [u,u,w,w,w,ham,u,u,w,w,w]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, -4], [11, 12, -3],
            [13, 14, 0], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, -1], [17, 16, -2], [15, 14, -5]]
order = [4,7,5,6,12,11,3,1,2,17,16,8,10,9,14,13,15]
open_order = [-1,-2,-5,-4,-3,0]
tensor_out, _, cost = ncon(tensors, connects, order=order, 
                           open_order=open_order, return_info=True)
tensor_ex = np.eye(64).reshape(4,4,4,4,4,4)

assert np.allclose(tensor_out, tensor_ex), (
    'incorrect network product: {n0} vs {n1}'.format(
        n0=tensor_out, n1=tensor_ex))

# partial trace and outer product
A = np.eye(8).reshape(2,2,2,2,2,2)
B = np.eye(16).reshape(2,2,2,2,2,2,2,2)
tensors = [A,B]
connects = [[-1,1,-2,-4,1,-5],[2,3,4,-3,2,3,4,-6]]
tensor_out = ncon(tensors, connects)
tensor_ex = np.eye(8).reshape(2,2,2,2,2,2) * 16

assert np.allclose(tensor_out, tensor_ex), (
    'incorrect network product: {n0} vs {n1}'.format(
        n0=tensor_out, n1=tensor_ex))

# unit test: partial_trace

labels = [1,1,2,3,4,3,5,2]
new_labels, tr_inds = partial_trace(labels)[:2]

new_labels_ex = np.array([4,5], dtype=int)
tr_inds_ex = np.array([1,2,3], dtype=int)

assert np.array_equal(new_labels, new_labels_ex), (
    'incorrect labels: {n0} vs {n1}'.format(
        n0=new_labels, n1=new_labels_ex))

assert np.array_equal(tr_inds_ex, tr_inds), (
    'incorrect traced inds: {n0} vs {n1}'.format(
        n0=tr_inds_ex, n1=tr_inds))

tensor = np.eye(16).reshape(2,2,2,2,2,2,2,2)
labels = [-1,1,2,-2,-3,1,2,-4]
tr_tensor, new_labels, tr_inds, cost = partial_trace(labels, tensor=tensor)

tr_tensor_ex = np.eye(4).reshape(2,2,2,2) * 4
cost_ex = 64
new_labels_ex = np.array([-1,-2,-3,-4], dtype=int)
tr_inds_ex = np.array([1,2], dtype=int)

assert np.allclose(tr_tensor, tr_tensor_ex), (
    'incorrect traced tensor')

assert np.array_equal(new_labels, new_labels_ex), (
    'incorrect labels: {n0} vs {n1}'.format(
        n0=new_labels, n1=new_labels_ex))

assert np.array_equal(tr_inds_ex, tr_inds), (
    'incorrect traced inds: {n0} vs {n1}'.format(
        n0=tr_inds_ex, n1=tr_inds))

assert cost == cost_ex, (
    'incorrect cost of partial trace: {n0} vs {n1}'.format(
        n0=cost, n1=cost_ex))

# unit test: `xcon`

# check against `remove_tensor` in conjunction with `ncon`
u = np.random.rand(2,2,2,2)
w = np.random.rand(2,2,4)
ham = np.random.rand(2,2,2,2,2,2)
rho = np.eye(64).reshape(4,4,4,4,4,4)
tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, 21], [11, 12, 22],
            [13, 14, 23], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, 18], [17, 16, 19], [15, 14, 20], 
            [18, 19, 20, 21, 22, 23]]
order = [4,7,5,6,12,11,3,1,2,17,16,8,10,9,14,13,15,18,19,20,21,22,23]
which_envs = [0,4,2,9,6,5,1,3]
all_envs, order, cost = xcon(tensors, connects, order=order, return_info=True,
                             which_envs=which_envs)

check_envs = []
for env in which_envs:
  temp_connects, temp_order = remove_tensor(connects, env, order=order)[:2]
  tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
  temp_tensors = [tensors[k] for k in range(len(tensors)) if k != env]
  temp_env = ncon(temp_tensors, temp_connects, order=temp_order)
  check_envs.append(temp_env)

is_matching = []
for k in range(len(which_envs)):
  is_matching.append(np.allclose(all_envs[k], check_envs[k]))

assert np.alltrue(is_matching), (
    'incorrect evaluation of tensor environments')

# test with evaluate optimal contraction order and compare cost
u = np.random.rand(4,4,3,3)
w = np.random.rand(3,3,4)
ham = np.random.rand(4,4,4,4,4,4)
rho = np.eye(64).reshape(4,4,4,4,4,4)
tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, 21], [11, 12, 22],
            [13, 14, 23], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, 18], [17, 16, 19], [15, 14, 20], 
            [18, 19, 20, 21, 22, 23]]
which_envs = [0,4,2,9,6,5,1,3]
all_envs, order, cost = xcon(tensors, connects, return_info=True,
                             which_envs=which_envs, solver='full')

dims = [tensor.shape for tensor in tensors] 
bn_costs = compute_costs(connects, order=order, dims=dims)
cost0 = sum(bn_costs[:(len(bn_costs)-1)])

assert cost == cost0, (
    'incorrect costs: {n0} vs {n1}'.format(
        n0=cost, n1=cost0))

# test no environment
u = np.random.rand(4,4,3,3)
w = np.random.rand(3,3,4)
ham = np.random.rand(4,4,4,4,4,4)
rho = np.eye(64).reshape(4,4,4,4,4,4)
tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, 21], [11, 12, 22],
            [13, 14, 23], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, 18], [17, 16, 19], [15, 14, 20], 
            [18, 19, 20, 21, 22, 23]]
scalar_out, order, cost = xcon(tensors, connects, return_info=True, 
                               solver='full')

scalar_out0, _, cost0 = ncon(tensors, connects, return_info=True, order=order)

assert cost == cost0, (
    'incorrect costs: {n0} vs {n1}'.format(
        n0=cost, n1=cost0))

assert scalar_out == scalar_out0, (
    'incorrect costs: {n0} vs {n1}'.format(
        n0=scalar_out, n1=scalar_out0))

# test single environment
u = np.random.rand(2,2,2,2)
w = np.random.rand(2,2,2)
ham = np.random.rand(2,2,2,2,2,2)
rho = np.eye(8).reshape(2,2,2,2,2,2)
tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, 21], [11, 12, 22],
            [13, 14, 23], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, 18], [17, 16, 19], [15, 14, 20], 
            [18, 19, 20, 21, 22, 23]]
order = [4,7,5,6,17,8,14,2,16,18,21,1,3,20,23,9,10,13,15,19,11,12,22]
which_envs = 4
tensor_out = xcon(tensors, connects, order=order, which_envs=which_envs)

temp_connects, temp_order = remove_tensor(connects, which_envs, order=order)[:2]
tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
temp_tensors = [tensors[k] for k in range(len(tensors)) if k != which_envs]
tensor_out0 = ncon(temp_tensors, temp_connects, order=temp_order)

tensors = [u,u,w,w,ham,u,u,w,w,w,rho]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, 21], [11, 12, 22],
            [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, 18], [17, 16, 19], [15, 14, 20], 
            [18, 19, 20, 21, 22, 23]]
open_order = [13, 14, 23]
tensor_out1 = xcon(tensors, connects, open_order=open_order, solver='full')

assert (np.allclose(tensor_out, tensor_out0) and 
        np.allclose(tensor_out0, tensor_out1)), (
            'incorrect network contraction')

# test symbolic defined networks
u = np.random.rand(2,2,2,2)
w = np.random.rand(2,2,2)
ham = np.random.rand(2,2,2,2,2,2)
rho = np.eye(8).reshape(2,2,2,2,2,2)
tensors = [u,u,w,w,w,ham,u,u,w,w,w]
connects = [[1, 3, 10, 11], [4, 7, 12, 13], [8, 10, -3], [11, 12, -4],
            [13, 14, -5], [2, 5, 6, 3, 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, -0], [17, 16, -1], [15, 14, -2]]
all_envs, order, cost = xcon(tensors, connects, return_info=True,
                             solver='full', standardize_inputs=False)

tensors = [u,u,w,w,w,ham,u,u,w,w,w]
connects = [[1, 'c', 'b', 11], [4, 7, 12, 13], [8, 'b', -3], [11, 12, -4],
            [13, 14, 'a'], [2, 5, 6, 'c', 4, 7], [1, 2, 9, 17], 
            [5, 6, 16, 15], [8, 9, -0], [17, 16, 'd'], [15, 14, -2]]
open_order = [-0, 'd', -2, -3, -4, 'a']
all_envs0, order0, cost0 = xcon(tensors, connects, return_info=True,
                             solver='full', open_order=open_order)

assert np.allclose(all_envs, all_envs0), (
            'incorrect network contraction')