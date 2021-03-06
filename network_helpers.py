# -*- coding: utf-8 -*-
"""network_helpers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rYDv3BiMRphBCWphVWX72zuAXc_SxnIg
"""

import numpy as np
from numpy import linalg as LA
from typing import Optional, List, Union, Tuple

"""
network_helpers

- expand_dims
- tprod
- matricize
- orthogonalize
"""

def intersect_lists(listA, listB):
  """ 
  Intersection of two lists, positions of intersecting elements and exclusive
  elements. 
  """  
  locsA = [k for k, ele in enumerate(listA) if ele in listB]
  common = [listA[k] for k in locsA]
  locsB = [listB.index(ele) for ele in common if ele in common]  
  exclusive = [ele for ele in (listA + listB) if ele not in common]
    
  return list(common), list(locsA), locsB, exclusive

def expand_dims(tensor, new_dims):
  """ 
  Expand the dims of a tensor by padding with zeros.
  """
  old_dims = tensor.shape
  dim_expand = [(0, max(new_dims[k] - old_dims[k],0)) for 
                k in range(tensor.ndim)]
  
  return np.pad(tensor, dim_expand)

def mkron(*mats):
  """ 
  Multi-kron: extends the functionality of numpy kron to accept a list of 
  arbitrarily many matrices.
  """
  final_mat = 1
  for mat in mats:
    final_mat = np.kron(final_mat, mat)
  return final_mat

def tprod(*tensor_list, do_matricize=True):
  """ 
  Tensor product for operators. Expands the functionality of `kron` to accept 
  tensors rather than just matrices, and to accept and arbitrary number of 
  inputs. The index ordering is defined such that if the inputs are Hermtian 
  matrices the the output tensor can be reshaped into a Hermitian matrix.
  """
  
  # take kron of each input sequentially
  shapes_L = []
  shapes_R = []
  final_tensor = np.array(1.0, dtype=float)
  for tensor in tensor_list:
    shapes_L = shapes_L + list(tensor.shape[:(tensor.ndim//2)])
    shapes_R = shapes_R + list(tensor.shape[(tensor.ndim//2):])
    final_tensor = np.kron(final_tensor, matricize(tensor))

  if do_matricize:
    return final_tensor
  else:
    return final_tensor.reshape(shapes_L + shapes_R)

def matricize(tensor, partition=None):
  """ Matricize an input tensor across some left/right partition. """
  
  if partition is None:
    partition = tensor.ndim // 2

  size_L = np.prod(tensor.shape[:partition])
  size_R = np.prod(tensor.shape[partition:])
 
  return tensor.reshape(size_L, size_R)

def orthogonalize(tensor, partition=None):
  """ Orthogonalize an input tensor across some left/right partition. """

  tshape = tensor.shape
  ut, st, vt = LA.svd(matricize(tensor, partition=partition), 
                      full_matrices=False)

  return (ut @ vt).reshape(tshape)