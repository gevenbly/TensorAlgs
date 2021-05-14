
import numpy as np

def pre_ncon(tensor_dims, connects, order=None):
  """ 
  Identify the labels involved in each tensor contraction (either a partial 
  trace or a binary tensor contraction).
  """
  # build dictionary between original and canonical dims
  nml_dims, fwd_dim_dict, rev_dim_dict = make_cannon_dims(tensor_dims)

  # build dictionary between original and canonical labels
  nml_connects, fwd_dict, rev_dict, npos, nneg = make_cannon_connects(connects)

  # find canonical order
  if order is None:
    nml_order = np.arange(npos) + 1
  else:
    nml_order = np.array([fwd_dict[ele] for ele in order])

  # check validity of network
  check_inputs(nml_connects, nml_dims, nml_order, rev_dict, rev_dim_dict)

  # identify contraction indices
  pt_cont, bn_cont = identify_cont_labels(nml_connects, nml_order)

  # compute contraction costs
  pt_costs, bn_costs = compute_costs(nml_connects, nml_dims, rev_dim_dict)
    
  return pt_cont, bn_cont, pt_costs, bn_costs

def compute_costs(connects, dims, rev_dim_dict):
  """ 
  Identify the labels involved in each tensor contraction (either a partial 
  trace or a binary tensor contraction).
  """
  nml_connects = [ele for ele in connects]
  nml_dims = [ele for ele in dims]

  # partial trace costs
  pt_costs = []
  for pt_labs in pt_cont:
    for count, sublist in enumerate(nml_connects):
      pt_inds, pt_locs0, _ = np.intersect1d(sublist, pt_labs, return_indices=True)
      if len(pt_inds) > 0:
        sublist = np.delete(sublist, pt_locs0)
        cont_cost = np.delete(nml_dims[count], pt_locs0)
        _, pt_locs1, _ = np.intersect1d(sublist, pt_labs, return_indices=True)
        
        nml_connects[count] = np.delete(sublist, pt_locs1)
        nml_dims[count] = np.delete(cont_cost, pt_locs1)
        break
    
    pt_costs.append(cont_cost)

  # binary contraction costs
  bn_costs = []
  for bn_labs in bn_cont:
    locs = [ele for ele in range(len(nml_connects)) if 
            sum(nml_connects[ele] == bn_labs[0]) > 0]
    cont_many, A_cont, B_cont = np.intersect1d(
        nml_connects[locs[0]],
        nml_connects[locs[1]],
        assume_unique=True,
        return_indices=True)

    nml_connects.append(np.concatenate((
        np.delete(nml_connects[locs[0]], A_cont),
        np.delete(nml_connects[locs[1]], B_cont))))
    bn_costs.append(np.concatenate((
        np.delete(nml_dims[locs[0]], A_cont), nml_dims[locs[1]])))
    nml_dims.append(np.concatenate((
        np.delete(nml_dims[locs[0]], A_cont),
        np.delete(nml_dims[locs[1]], B_cont))))

    del nml_connects[locs[1]]
    del nml_connects[locs[0]]
    del nml_dims[locs[1]]
    del nml_dims[locs[0]]

  # tally the total partial trace costs
  is_symbolic = False
  int_pt_costs = []
  fin_pt_costs = []
  for cost in pt_costs:
    uni_dims = np.unique(cost)

    str_cost = ''
    int_cost = 1
    for dim in uni_dims:
      degen = sum(cost == dim)
      value = rev_dim_dict[dim]

      if isinstance(value, str):
        str_cost += '(' + value + '^' + str(degen) + ')'
        is_symbolic = True
      elif isinstance(value, int):
        int_cost = int_cost * value**degen

    int_pt_costs.append(int_cost)
    fin_pt_costs.append(str(int_cost) + '*' + str_cost)

  # tally the total binary contraction costs
  int_bn_costs = []
  fin_bn_costs = []
  for cost in bn_costs:
    uni_dims = np.unique(cost)

    str_cost = ''
    int_cost = 1
    for dim in uni_dims:
      degen = sum(cost == dim)
      value = rev_dim_dict[dim]

      if isinstance(value, str):
        str_cost += '(' + value + '^' + str(degen) + ')'
        is_symbolic = True
      elif isinstance(value, int):
        int_cost = int_cost * value**degen

    int_bn_costs.append(int_cost)
    fin_bn_costs.append(str(int_cost) + '*' + str_cost)

  if not is_symbolic:
    fin_pt_costs = int_pt_costs
    fin_bn_costs = int_bn_costs

  return fin_pt_costs, fin_bn_costs

def identify_cont_labels(connects, order):
  """ 
  Identify the labels involved in each tensor contraction (either a partial 
  trace or a binary tensor contraction).
  """

  nml_order = [ele for ele in order]
  nml_connects = [ele for ele in connects]

  # indentify partial trace indices to be contracted
  pt_cont = []
  for count, sublist in enumerate(nml_connects):
    uni_labs, uni_locs = np.unique(sublist, return_index=True)
    uni_dims = [tensor_dims[count][loc] for loc in uni_locs]
    num_cont = len(sublist) - len(uni_labs)
    if num_cont > 0:
      dup_list = []
      for ele in uni_labs:
        temp_locs = np.where(sublist == ele)[0]
        if len(temp_locs) == 2:
          dup_list.append(ele)
          sublist = np.delete(sublist, temp_locs)
          nml_order = np.delete(nml_order, nml_order==ele)
      
      pt_cont.append(np.array(dup_list))
      nml_connects[count] = sublist

  # indentify binary contraction indices 
  bn_cont = []
  while len(nml_order) > 0:
    locs = [ele for ele in range(len(nml_connects)) 
            if sum(nml_connects[ele] == nml_order[0]) > 0]

    cont_many, A_cont, B_cont = np.intersect1d(
        nml_connects[locs[0]],
        nml_connects[locs[1]],
        assume_unique=True,
        return_indices=True)
    
    bn_cont.append(cont_many)
    nml_connects.append(np.concatenate((
      np.delete(nml_connects[locs[0]], A_cont),
      np.delete(nml_connects[locs[1]], B_cont))))
    del nml_connects[locs[1]]
    del nml_connects[locs[0]]
    nml_order = np.delete(nml_order, np.intersect1d(nml_order, 
                                                    cont_many, 
                                                    return_indices=True)[1])
    
  return pt_cont, bn_cont

def make_cannon_dims(dims):
  """ 
  Create dict holding the unique tensor dims, which may be input either as 
  strings or integers, and transform the dims according to this dict. 
  """
  
  # flatten the list of connections
  flat_dims = [item for sublist in dims for item in sublist]

  # find unique entries
  uni_dims = []
  for ele in flat_dims:
    if ele not in uni_dims:
      uni_dims.append(ele)
  
  # create dictionary to map between original and cannonical dims
  fwd_dict = dict(zip(uni_dims, np.arange(len(uni_dims))))
  rev_dict = dict(zip(np.arange(len(uni_dims)), uni_dims))

  # make canonical dims
  can_dims = []
  for tensor in dims:
    temp_dims = []
    for lab in tensor:
      temp_dims.append(fwd_dict[lab])
    can_dims.append(np.array(temp_dims, dtype=int))

  return can_dims, fwd_dict, rev_dict

def make_cannon_connects(connects):
  """
  Takes in a set of `connects` defining a network, where index labels can be
  given either as `int` or `str` and returns dicts mapping between cannonical
  labels: where open (external) indices are labelled with negative integers 
  (starting at -1) and closed (internal) indices are labelled with positive
  integers (starting at +1). Sorting of indices is done alpha-numerically.
  """

  # flatten the list of connections
  flat_connects = [item for sublist in connects for item in sublist]

  # separate ints from strs
  int_connects = []
  str_connects = []
  for ele in flat_connects:
    if isinstance(ele, int):
      int_connects.append(ele)
    elif isinstance(ele, str):
      str_connects.append(ele)

  # separate single (open) indices from double (closed) indices
  sgl_str = []
  dbl_str = []
  for ele in str_connects:
    if str_connects.count(ele) == 1:
      sgl_str.append(ele)
    elif str_connects.count(ele) == 2:
      if dbl_str.count(ele) == 0:
        dbl_str.append(ele)
    else:
      raise ValueError("index label {ind} is repeated more than twice".format(
          ind = "`" + ele + "`"))

  sgl_int = []
  dbl_int = []
  for ele in int_connects:
    if int_connects.count(ele) == 1:
      sgl_int.append(ele)
    elif int_connects.count(ele) == 2:
      if dbl_int.count(ele) == 0:
        dbl_int.append(ele)
    else:
      raise ValueError("index label {ind} is repeated more than twice".format(
          ind = "`" + str(ele) + "`"))
  
  # sort and combine index labels
  sgl_str.sort()
  dbl_str.sort()
  sgl_int.sort()
  sgl_int.reverse()
  dbl_int.sort()
  open_inds = sgl_str + sgl_int
  clsd_inds = dbl_str + dbl_int
  num_neg = len(open_inds)
  num_pos = len(clsd_inds)
  
  # create dictionary to map between original and cannonical labels
  pos_labs = dict(zip(open_inds, -np.arange(1,len(open_inds) + 1)))
  neg_labs = dict(zip(clsd_inds, np.arange(1,len(clsd_inds) + 1)))
  can_labs = {**pos_labs, **neg_labs}
  rev_can_labs = dict(zip(can_labs.values(), can_labs.keys()))

  # make canonical connections
  can_connects = []
  for tensor in connects:
    temp_inds = []
    for lab in tensor:
      temp_inds.append(can_labs[lab])
    can_connects.append(np.array(temp_inds, dtype=int))

  return can_connects, can_labs, rev_can_labs, num_pos, num_neg

def check_inputs(connects, dims, con_order, rev_dict, rev_dim_dict):
  """ Check consistancy of NCON inputs"""

  flat_connect = np.concatenate(connects)
  pos_ind = flat_connect[flat_connect > 0]
  neg_ind = flat_connect[flat_connect < 0]

  # check that lengths of lists match
  if len(dims) != len(connects):
    raise ValueError((
        'Network definition error: mismatch between {n0} tensors given but {n1}'
        ' index sublists given'.format(n0 = str(len(dims)), 
                                       n1 = str(len(connects)))))

  # check that tensors have the right number of indices
  for ele in range(len(dims)):
    if len(dims[ele]) != len(connects[ele]):
      raise ValueError(
          'Network definition error: number of indices does not match number'
          ' of labels on tensor {n0}: {n1}-indices versus {n2}-labels'.format(
              n0 = str(ele),
              n1 = str(len(dims[ele])),
              n2 = str(len(connects[ele]))))

  # check that contraction order is valid
  if not np.array_equal(np.sort(con_order), np.unique(pos_ind)):
    print(np.sort(con_order))
    print(np.unique(pos_ind))
    raise ValueError('Network definition error: invalid contraction order')

  # check that positive indices are valid and contracted tensor dimensions match
  flat_dims = np.array([item for sublist in dims for item in sublist])
  for ind in np.unique(pos_ind):
    if sum(pos_ind == ind) == 1:
      raise ValueError(
        'Network definition error: only one index labelled {n0}'
        .format(n0 = "`" + str(rev_dict[ind]) + "`"))
    elif sum(pos_ind == ind) > 2:
      raise ValueError(
        'Network definition error: more than two indices labelled {n0}'
        .format(n0 = "`" + str(rev_dict[ind]) + "`"))

    cont_dims = flat_dims[flat_connect == ind]
    if cont_dims[0] != cont_dims[1]:
      raise ValueError(
          'Network definition error: tensor dimension mismatch on'
          ' index labelled {n0}: dim-{n1} versus dim-{n2}'
          .format(n0 = "`" + str(rev_dict[ind]) + "`", 
                  n1 = str(rev_dim_dict[cont_dims[0]]), 
                  n2 = str(rev_dim_dict[cont_dims[1]])))

  return True
