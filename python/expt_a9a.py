from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
# import matplotlib.pyplot as plt

import copy
from multiprocessing import Pool

import time
import os, os.path as osp
import csv
import cPickle as pick
# import sqlparse as sql

import activeSearchInterface as ASI
import competitorsInterface as CI
import gaussianRandomFeatures as GRF

import data_utils as du
import graph_utils as gu

import lapsvmp as LapSVM
import anchorGraph as AG

import IPython

data_dir = osp.join(du.data_dir, 'aaai17_data')
results_dir = osp.join(du.results_dir, 'aaai17/a9a/expts')


def test_a9a (arg_dict):

  if 'seed' in arg_dict:
    seed = arg_dict['seed']
  else: seed = None
  
  if 'prev' in arg_dict:
    prev = arg_dict['prev']
  else: prev = 0.05

  if 'save' in arg_dict:
    save = arg_dict['save']
  else: save = False

  verbose = True
  sparse = False
  eta = 0.5
  K = 200
  
  t1 = time.time()
  X0,Y0 = du.load_a9a()
  X_norms = np.sqrt((X0*X0).sum(axis=0)).squeeze()
  X0 = (X0/X_norms) # Normalization
  # X0,W = du.whiten_data(X0, sparse=sparse, rtn_W=True, thresh=1e-6)
  print('Time taken to load a9a data: %.2f'%(time.time()-t1))

  t1 = time.time()
  ag_file = osp.join(data_dir, 'a9a_AG_kmeans_500.npz')
  Z,rL = AG.load_AG(ag_file)
  print ('Time taken to load mnist AG: %.2f'%(time.time()-t1))
  
  nr.seed(prev_seed)

  # Changing prevalence of +
  if Y0.sum()/Y0.shape[0] < prev:
    prev = Y0.sum() / Y0.shape[0]
    X, Y = X0, Y0
  else:
    t1 = time.time()
    X, Y, inds = du.change_prev (X0, Y0, prev=prev, return_inds=True)
    Z = Z[inds, :]
    print ('Time taken to change prev: %.2f'%(time.time()-t1))

  strat_frac = 1.0
  if strat_frac < 1.0:
    t1 = time.time()
    X, Y, strat_inds = du.stratified_sample(X, Y, classes=[0,1], strat_frac=strat_frac,return_inds=True)
    Z = Z[strat_inds, :]
    print ('Time taken to stratified sample: %.2f'%(time.time()-t1))
  d,n = X.shape

  nr.seed(seed)
  # init points
  n_init = 1
  init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_init,replace=False)]
  init_labels = {p:1 for p in init_pt}

  t1 = time.time()
  # Kernel AS
  pi = Y.sum() * 1.0 / Y.shape[0]
  alpha = 1e-6
  ASprms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta, alpha=alpha)
  kAS = ASI.linearizedAS (ASprms)
  kAS.initialize(X, init_labels=init_labels)
  print ('KAS initialized.')
  
  # NN AS
  normalize = True
  NNprms = ASI.WNParameters(normalize=normalize ,sparse=sparse, verbose=verbose)
  NNAS = ASI.weightedNeighborAS (NNprms)
  NNAS.initialize(X, init_labels=init_labels)
  print ('NNAS initialized.')

  # # # anchorGraph AS
  gamma = 0.01
  AGprms = CI.anchorGraphParameters(gamma=gamma, sparse=sparse, verbose=verbose)
  AGAS = CI.anchorGraphAS (AGprms)
  AGAS.initialize(Z, rL, init_labels=init_labels) 
  print ('AGAS initialized.')

  hits_K = [n_init]
  hits_NN = [n_init]
  hits_AG = [n_init]

  print ('Time taken to initialize all approaches: %.2f'%(time.time()-t1))
  print ('Beginning experiment.')

  import IPython
  IPython.embed()

  for i in xrange(K):
    print('Iter %i out of %i'%(i+1,K))
    idx1 = kAS.getNextMessage()
    kAS.setLabelCurrent(Y[idx1])
    hits_K.append(hits_K[-1]+Y[idx1])

    idx2 = NNAS.getNextMessage()
    NNAS.setLabelCurrent(Y[idx2])
    hits_NN.append(hits_NN[-1]+Y[idx2])

    #idx3 = LapSVMAS.getNextMessage()
    #LapSVMAS.setLabelCurrent(Y[idx3])
    #hits_LSVM.append(hits_LSVM[-1]+Y[idx3])

    idx4 = AGAS.getNextMessage()
    AGAS.setLabelCurrent(Y[idx4])
    hits_AG.append(hits_AG[-1]+Y[idx4])

    print('')
  
  if save:
    if seed is None: 
      seed = -1
    pred_results = {'kAS': [kAS.f, kAS.unlabeled_idxs],
                    'NNAS': [NNAS.f, NNAS.unlabeled_idxs],
                    'AGAS': [AGAS.f, AGAS.unlabeled_idxs],
                    'Y': Y}
    save_results = {'kAS': hits_K,
            'NNAS': hits_NN,
            'AGAS': hits_AG}

    fname = 'expt_seed_%d.cpk'%seed
    fname_res = 'res_expt_seed_%d.cpk'%seed
    dname = osp.join(results_dir, 'rbf/%.2f/'%(prev*100))
    if not osp.isdir(dname):
      os.makedirs(dname)

    # fname = osp.join(dname,fname)
    # with open(fname, 'w') as fh: pick.dump({'res': save_results, 'pred': pred_results}, fh)
    with open(osp.join(dname, fname), 'w') as fh:
      pick.dump(pred_results, fh)
    with open(osp.join(dname, fname_res), 'w') as fh:
      pick.dump(save_results, fh)
  else:
    IPython.embed()


def test_a9a_imf (arg_dict):

  if 'seed' in arg_dict:
    seed = arg_dict['seed']
  else: seed = None
  
  if 'prev' in arg_dict:
    prev = arg_dict['prev']
  else: prev = 0.05

  if 'save' in arg_dict:
    save = arg_dict['save']
  else: save = False

  verbose = True
  sparse = False
  eta = 0.5
  K = 200
  
  t1 = time.time()
  X0,Y0 = du.load_a9a()
  X_norms = np.sqrt((X0*X0).sum(axis=0)).squeeze()
  X0 = (X0/X_norms) # Normalization
  # X0,W = du.whiten_data(X0, sparse=sparse, rtn_W=True, thresh=1e-6)
  print('Time taken to load a9a data: %.2f'%(time.time()-t1))
  
  nr.seed(prev_seed)
  # Changing prevalence of +
  if Y0.sum()/Y0.shape[0] < prev:
    prev = Y0.sum() / Y0.shape[0]
    X, Y = X0, Y0
  else:
    t1 = time.time()
    X, Y, inds = du.change_prev (X0, Y0, prev=prev, return_inds=True)
    print ('Time taken to change prev: %.2f'%(time.time()-t1))

  strat_frac = 1.0
  if strat_frac < 1.0:
    t1 = time.time()
    X, Y, strat_inds = du.stratified_sample(X, Y, classes=[0,1], strat_frac=strat_frac,return_inds=True)
    print ('Time taken to stratified sample: %.2f'%(time.time()-t1))
  d,n = X.shape

  nr.seed(seed)
  # init points
  n_init = 1
  init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_init,replace=False)]
  init_labels = {p:1 for p in init_pt}

  alphas = [0.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
  t1 = time.time()
  # Kernel AS
  pi = Y.sum() * 1.0 / Y.shape[0]
  kAS_dict = {}
  hits_dict = {}
  for alpha in alphas:
    ASprms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta, alpha=alpha)
    kAS = ASI.linearizedAS (ASprms)
    kAS.initialize(X, init_labels=init_labels)
    kAS_dict[alpha] = kAS
    hits_dict[alpha] = [n_init]
  print ('Time taken to initialize all: %.2f'%(time.time()-t1))
  print ('Beginning experiment.')

  # import IPython
  # IPython.embed()

  for i in xrange(K):
    for alpha in alphas:
      print(alpha)
      idx = kAS_dict[alpha].getNextMessage()
      kAS_dict[alpha].setLabelCurrent(Y[idx])
      hits_dict[alpha].append(hits_dict[alpha][-1]+Y[idx])
    print('')
  
  if save:
    if seed is None: 
      seed = -1
    pred_results = {
        alpha: [kAS_dict[alpha].f, kAS_dict[alpha].unlabeled_idxs]
        for alpha in alphas
    }
    save_results = hits_dict

    fname = 'expt_seed_%d.cpk'%seed
    fname_res = 'res_expt_seed_%d.cpk'%seed
    dname = osp.join(results_dir, 'imf/%.2f/'%(prev*100))
    if not osp.isdir(dname):
      os.makedirs(dname)

    with open(osp.join(dname, fname), 'w') as fh:
      pick.dump(pred_results, fh)
    with open(osp.join(dname, fname_res), 'w') as fh:
      pick.dump(save_results, fh)
  else:
    IPython.embed()


def test_one_point(arg_dict):
  if 'seed' in arg_dict:
    seed = arg_dict['seed']
  else: seed = None
  
  if 'prev' in arg_dict:
    prev = arg_dict['prev']
  else: prev = 0.05

  if 'save' in arg_dict:
    save = arg_dict['save']
  else: save = False

  verbose = True
  sparse = False
  eta = 0.5
  
  t1 = time.time()
  X0,Y0 = du.load_a9a()
  X_norms = np.sqrt((X0*X0).sum(axis=0)).squeeze()
  X0 = (X0/X_norms) # Normalization

  nr.seed(seed)
  
  # Changing prevalence of +
  if Y0.sum()/Y0.shape[0] < prev:
    prev = Y0.sum() / Y0.shape[0]
    RX, Y = X0, Y0
  else:
    t1 = time.time()
    RX, Y, inds = du.change_prev (X0, Y0, prev=prev, return_inds=True)
    print ('Time taken to change prev: %.2f'%(time.time()-t1))

  strat_frac = 1.0
  if strat_frac < 1.0:
    t1 = time.time()
    RX, Y, strat_inds = du.stratified_sample(RX, Y, classes=[0,1], strat_frac=strat_frac,return_inds=True)
    print ('Time taken to stratified sample: %.2f'%(time.time()-t1))
  d,n = RX.shape

  t1 = time.time()
  # Kernel AS
  pi = Y.sum() * 1.0 / Y.shape[0]
  alpha = 1e-6
  ASprms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta, alpha=alpha)
  kAS = ASI.linearizedAS (ASprms)
  kAS.initialize(RX)
  print ('KAS initialized.')
  
  # NN AS
  normalize = True
  NNprms = ASI.WNParameters(normalize=normalize ,sparse=sparse, verbose=verbose)
  NNAS = ASI.weightedNeighborAS (NNprms)
  NNAS.initialize(RX)
  print ('NNAS initialized.')

  num_trials = 100
  kAS_pred = 0
  NNAS_pred = 0
  nz_ct = 0

  del X0
  # init points
  Ynz = Y.nonzero()[0]
  Yz = (Y==0).nonzero()[0]
  init_ppts = Ynz[nr.permutation(len(Ynz))]
  init_npts = Yz[nr.choice(len(Ynz), len(Ynz), replace=True)]
  # init_ppts = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), num_trials, replace=False)]
  # init_npts = (Y == 0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), num_trials, replace=True)]

  IPython.embed()

  itr = 0
  for ppt, npt in zip(init_ppts, init_npts):
    itr += 1
    print('Iter: %i'%itr)
    kASc = copy.deepcopy(kAS)
    kASc.setLabel(ppt, 1)
    kASc.setLabel(npt, 0)
    next_inds1 = [kASc.unlabeled_idxs[idx] for idx in (-kASc.f[kASc.unlabeled_idxs]).argsort()[:100]]
    del kASc

    nASc = copy.deepcopy(NNAS)
    nASc.setLabel(ppt, 1)
    nASc.setLabel(npt, 0)
    next_inds2 = [nASc.unlabeled_idxs[idx] for idx in (-nASc.f).argsort()[:100]]
    del nASc

    v1 = Y[next_inds1].sum()
    v2 = Y[next_inds2].sum()

    print(v1, v2)
    kAS_pred += v1
    NNAS_pred += v2
    if v1 + v2 > 0: nz_ct += 1
    print(kAS_pred, NNAS_pred, nz_ct)

    if nz_ct >= num_trials:
      break
    

  IPython.embed()

                            
if __name__ == '__main__':
  import sys

  ## Argument 1: 1/2 -- rbf/imf expt
  ## Argument 2: number of experiments to run in parallel
  ## Argument 3: prevalence of +ve class

  exp_type = 1
  num_expts = 3
  prev = 0.05

  if len(sys.argv) > 1:
    try:
      exp_type = int(sys.argv[1])
    except:
      exp_type = 1
    if exp_type not in [1,2,3]:
      exp_type = 1

  if len(sys.argv) > 2:
    try:
      num_expts = int(sys.argv[2])
    except:
      num_expts = 3
    if num_expts > 10:
      num_expts = 10
    elif num_expts < 1:
      num_expts = 1

  if len(sys.argv) > 3:
    try:
      prev = float(sys.argv[3])
    except:
      prev = 0.05
    if prev < 0 or prev > 0.05:
      prev = 0.05

  test_funcs = {1:test_a9a, 2:test_a9a_imf, 3:test_one_point}


  # nr.seed(int((time.clock()%(0.01))*10e6))
  # rerunning crashed expt
  # seeds = [9487273, 1409143, 4861323, 1226788, 5698172, 3257080, 8462562, 5039904, 8008923, 9724002]
  prev_seed = nr.randint(10e6)
  seeds = nr.choice(int(10e6),num_expts,replace=False)
  print prev_seed
  print seeds
  # seeds = range(1, num_expts+1)
  save = (num_expts != 1)
  arg_dicts = [{'prev':prev, 'seed':s, 'save':save} for s in seeds]

  if num_expts == 1:
    print ('Running 1 experiment')
    #test_funcs[dset](seeds[0])
    test_funcs[exp_type](arg_dicts[0])
  else:
    print ('Running %i experiments'%num_expts)
    pl = Pool(num_expts)
    pl.map(test_funcs[exp_type], arg_dicts)