#! /usr/bin/python
from __future__ import division
import time
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.linalg as slg, scipy.spatial.distance as ssd, scipy.sparse as ss
import scipy.io as sio
import matplotlib.pyplot as plt, matplotlib.cm as cm

import IPython # debugging

import activeSearchInterface as ASI
import adaptiveActiveSearch as AAS
import similarityLearning as SL
import dataUtils as du

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def polarToCartesian (r, theta):
  return r*np.array([np.cos(theta), np.sin(theta)])

def cartesianToPolar (x, y):
  return np.array([nlg.norm([x,y]), np.arctan(y,x)])

# def createSwissRolls (npts = 500, prev = 0.5, c = 1.0, nloops = 1.5, var = 0.05, shuffle=False):
#   # npts    -- number of points overall
#   # prev    -- prevalence of positive class
#   # c     -- r = c*theta
#   # nloops  -- number of loops of swiss roll
#   # var     -- variance of 0-mean gaussian noise along the datapoints
#   # shuffle -- shuffle points or keep them grouped as 1/0

#   std = np.sqrt(var)
#   n1 = int(prev*npts);
#   n2 = npts-n1

#   angle_range1 = np.linspace(np.pi/2, 2*nloops*np.pi, n1)
#   angle_range2 = np.linspace(np.pi/2, 2*nloops*np.pi, n2)

#   X = np.empty([npts,2])
#   Y = np.array(n1*[1] + n2*[0])

#   for i in xrange(n1):
#     a = angle_range1[i]
#     X[i,:] = polarToCartesian(a*c, a) + nr.randn(1,2)*std
#   for i in xrange(n2):
#     a = angle_range2[i]
#     X[i+n1,:] = polarToCartesian(a*c, a+np.pi) + nr.randn(1,2)*std

#   if shuffle:
#     shuffle_inds = nr.permutation(npts)
#     X = X[shuffle_inds,:]
#     Y = Y[shuffle_inds]

#   return X,Y


def createSwissRolls (npts = 500, prev = 0.5, c = 1.0, nloops = 1.5, var = 0.05, var2 = None, shuffle=False):
  # npts    -- number of points overall
  # prev    -- prevalence of positive class
  # c     -- r = c*theta
  # nloops  -- number of loops of swiss roll
  # var     -- variance of 0-mean gaussian noise along the datapoints
  # shuffle -- shuffle points or keep them grouped as 1/0

  std1 = np.sqrt(var)
  if var2 is None:
    var2 = var
  std2 = np.sqrt(var2)
  n1 = int(prev*npts);
  n2 = npts-n1

  angle_range1 = np.linspace(np.pi/2, 2*nloops*np.pi, n1)
  angle_range2 = np.linspace(np.pi/2, 2*nloops*np.pi, n2)

  X = np.empty([npts,2])
  Y = np.array(n1*[1] + n2*[0])

  for i in xrange(n1):
    a = angle_range1[i]
    X[i,:] = polarToCartesian(a*c, a) + nr.randn(1,2)*std1
  for i in xrange(n2):
    a = angle_range2[i]
    X[i+n1,:] = polarToCartesian(a*c, a+np.pi) + nr.randn(1,2)*std2

  if shuffle:
    shuffle_inds = nr.permutation(npts)
    X = X[shuffle_inds,:]
    Y = Y[shuffle_inds]

  return X,Y

def plotData(X, Y, f=None, labels=None, thresh=None, block=False, fid=None):

  if fid is not None:
    fig = plt.figure(fid)
  plt.clf()

  if f is None:
    pos_inds = (Y==1).nonzero()[0]
    neg_inds = (Y==0).nonzero()[0]

    plt.scatter(X[pos_inds,0], X[pos_inds,1], color='b', label='positive')
    plt.scatter(X[neg_inds,0], X[neg_inds,1], color='r', label='negative')
    
  else:
    # assert thresh is not None
    assert labels is not None

    pos_inds = (labels==1).nonzero()[0]
    plt.scatter(X[pos_inds,0], X[pos_inds,1], color='b', label='positive', marker='x', linewidth=2)
    neg_inds = (labels==0).nonzero()[0] 
    plt.scatter(X[neg_inds,0], X[neg_inds,1], color='r', label='negative', marker='x', linewidth=2)

    plt.set_cmap('RdBu')
    rest_inds = (labels==-1).nonzero()[0]
    colors = cm.RdBu(f)
    plt.scatter(X[rest_inds,0], X[rest_inds,1], color=colors, label='unlabeled', linewidth=1)

  if fid is not None:
    plt.title('ID: %i'%fid)
  # plt.legend()
  plt.show(block=block)
  plt.pause(0.001)
  # time.sleep(0.5)

def createEpsilonGraph (X, eps=1, kernel='rbf', gamma=1):
  ## Creates an epsilon graph as follows:
  ## create with edges between points with distance < eps
  ## edge weights are given by kernel
  
  if kernel not in ['rbf']:
    raise NotImplementedError('This function does not support %s kernel.'%kernel)

  dists = ssd.cdist(X,X)
  eps_neighbors = dists < eps

  if kernel == 'rbf':
    A = eps_neighbors*np.exp(-gamma*dists)

  return A

def testSwissRolls ():
  npts = 500
  prev = 0.5
  c = 1.0
  nloops = 1.5
  var = 0.05
  shuffle = False
  
  X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)
  plotData(X,Y)


def testNaiveAS ():

  ## Create swiss roll data
  npts = 1000
  prev = 0.5
  c = 0.2
  nloops = 1.5
  var = 0.012
  shuffle = False
  eps = 0.2
  gamma = 10
  
  X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)
  A = createEpsilonGraph (X, eps=eps, gamma=gamma)

  ## Initialize naiveAS
  pi = prev
  eta = 0.5
  w0 = None
  sparse = False
  verbose = True
  prms = ASI.Parameters(pi=pi, eta=eta, w0=w0, sparse=sparse, verbose=verbose)

  np_init = 1
  nn_init = 1
  n_init = np_init + nn_init
  initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
  initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
  init_labels = {p:1 for p in initp_pt}
  for p in initn_pt: init_labels[p] = 0

  nAS = ASI.naiveAS (prms)
  nAS.initialize(A, init_labels)

  plotData(X, None, nAS.f[(nAS.labels==-1).nonzero()[0]], nAS.labels)

  hits = [n_init]
  K = 200
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx = nAS.getNextMessage()
    nAS.setLabelCurrent(Y[idx])
    hits.append(hits[-1]+Y[idx])

    plotData(X, None, nAS.f[(nAS.labels==-1).nonzero()[0]], nAS.labels)

    print('')

  IPython.embed()

def testWNAS ():

  ## Create swiss roll data
  npts = 1000
  prev = 0.5
  c = 0.2
  nloops = 1.5
  var = 0.012
  shuffle = False
  eps = 2
  gamma = 10
  
  X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)
  A = createEpsilonGraph (X, eps=eps, gamma=gamma)

  ## Initialize naiveAS
  pi = prev
  sparse = False
  verbose = True
  normalize = True
  prms = ASI.WNParameters(sparse=sparse, verbose=verbose, normalize=normalize)

  np_init = 1
  nn_init = 1
  n_init = np_init + nn_init
  initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
  initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
  init_labels = {p:1 for p in initp_pt}
  for p in initn_pt: init_labels[p] = 0

  wnAS = ASI.weightedNeighborGraphAS (prms)
  wnAS.initialize(A, init_labels)

  plotData(X, None, (wnAS.f+1)/2, wnAS.labels)

  hits = [n_init]
  K = 200
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx = wnAS.getNextMessage()
    wnAS.setLabelCurrent(Y[idx])
    hits.append(hits[-1]+Y[idx])

    plotData(X, None, (wnAS.f+1)/2, wnAS.labels)

    print('')

  IPython.embed()

def testRWNAS ():

  ## Create swiss roll data
  npts = 1000
  prev = 0.5
  c = 1
  nloops = 1.5
  var = 0.2
  shuffle = False
  eps = 2
  gamma = 10
  
  X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)
  A = createEpsilonGraph (X, eps=eps, gamma=gamma)

  ## Initialize naiveAS
  pi = prev
  sparse = False
  verbose = True
  lw = 1
  cut_connections = True
  prms = AAS.RWParameters(sparse=sparse, verbose=verbose, lw=lw, cut_connections=cut_connections)

  np_init = 1
  nn_init = 1
  n_init = np_init + nn_init
  initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
  initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
  init_labels = {p:1 for p in initp_pt}
  for p in initn_pt: init_labels[p] = 0

  rwnAS = AAS.reweightedNaiveAS (prms)
  rwnAS.initialize(A, init_labels)

  plotData(X, None, rwnAS.f, rwnAS.labels)

  hits = [n_init]
  K = 200
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx = rwnAS.getNextMessage()
    rwnAS.setLabelCurrent(Y[idx])
    hits.append(hits[-1]+Y[idx])

    plotData(X, None, rwnAS.f, rwnAS.labels)

    print('')

  IPython.embed()


def testMPCKAS ():

  ## Create swiss roll data
  npts = 1000
  prev = 0.5
  c = 1
  nloops = 1.5
  var = 0.2
  shuffle = False
  eps = 2
  gamma = 10
  
  X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)
  A = createEpsilonGraph (X, eps=eps, gamma=gamma)

  ## Initialize naiveAS
  pi = prev
  sparse = False
  verbose = True
  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi)

  np_init = 1
  nn_init = 1
  n_init = np_init + nn_init
  initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
  initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
  init_labels = {p:1 for p in initp_pt}
  for p in initn_pt: init_labels[p] = 0

  T = 20
  mpckAS = AAS.MPCKLinearizedAS (A0=None, T = T, ASparams=prms)
  mpckAS.initialize(X.T, init_labels)
  lAS = ASI.linearizedAS (prms)
  lAS.initialize(X.T, init_labels)

  plotData(X, None, mpckAS.kAS.f, mpckAS.kAS.labels)

  hits1 = [n_init]
  hits2 = [n_init]
  K = 200
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx1 = mpckAS.getNextMessage()
    idx2 = lAS.getNextMessage()
    mpckAS.setLabelCurrent(Y[idx1])
    lAS.setLabelCurrent(Y[idx2])
    hits1.append(hits1[-1]+Y[idx1])
    hits2.append(hits2[-1]+Y[idx2])

    plotData(X, None, mpckAS.kAS.f, mpckAS.kAS.labels)

    print('')

  IPython.embed()


def testNPKAS ():

  ## Create swiss roll data
  npts = 200
  prev = 0.5
  c = 1
  nloops = 1.5
  var = 0.2
  shuffle = False
  eps = 2
  gamma = 10
  
  X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)
  A = createEpsilonGraph (X, eps=eps, gamma=gamma)

  ## Initialize naiveAS
  pi = prev
  sparse = False
  verbose = True

  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi)

  np_init = 1
  nn_init = 1
  n_init = np_init + nn_init
  initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
  initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
  init_labels = {p:1 for p in initp_pt}
  for p in initn_pt: init_labels[p] = 0

  T = 1
  npkAS = AAS.NPKNaiveAS (T=T, ASparams=prms)
  npkAS.initialize(A, init_labels)
  lAS = ASI.naiveAS (prms)
  lAS.initialize(A, init_labels)

  plotData(X, None, npkAS.kAS.f, npkAS.kAS.labels)

  hits1 = [n_init]
  hits2 = [n_init]
  K = 200
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx1 = npkAS.getNextMessage()
    idx2 = lAS.getNextMessage()
    npkAS.setLabelCurrent(Y[idx1])
    lAS.setLabelCurrent(Y[idx2])
    hits1.append(hits1[-1]+Y[idx1])
    hits2.append(hits2[-1]+Y[idx2])

    plotData(X, None, npkAS.kAS.f, npkAS.kAS.labels)

    print('')

  IPython.embed()

def testAEW():

  import scipy.io as sio
  # Test script for AEW
  # (Results depend on random seeds)
  noise_level = 3# 1 or 2
  var1 = 0.001
  var2 = 0.001
  prev = 0.10

  # Parameters for AEW
  k = 10 # The number of neighbors in kNN graph
  sigma = 'median' # Kernel parameter heuristics 'median' or 'local-scaling'
  max_iter = 100
  param = SL.MSALPParameters(k=k, sigma=sigma, max_iter=max_iter)
  # --------------------------------------------------


  X,Y = du.generate_syndata(noise_level, nc=3, prev=prev, var1=var1, var2=var2, display=True)
  lb_idx = SL.select_labeled_nodes(Y,10)
  trY = np.zeros(Y.shape)
  trY[lb_idx,:] = Y[lb_idx,:]
  # data = sio.loadmat('tempdata.mat')
  # X = data['X'].T
  # Y = data['Y']
  # lb_idx = data['lb_idx']
  # trY = data['trY']
  
  print('Optimizing edge weights by AEW\n')
  W,W0 = SL.AEW(X,param)

  print('Estimating labels by harmonic Gaussian model ... ')
  L = np.diag(W.sum(axis=0)) - W
  F = SL.HGF(L,trY)
  L0 = np.diag(W0.sum(axis=0)) - W0
  F0 = SL.HGF(L0,trY)
  print('done\n')

  err_rate = SL.hamming_loss(Y,F) / (Y.shape[0] - len(lb_idx))
  err_rate0 = SL.hamming_loss(Y,F0) / (Y.shape[0] - len(lb_idx))

  print('[REPORT]\n')
  print('The number of classes = %d\n'%Y.shape[1])
  print('The number of nodes = %d\n'%Y.shape[0])
  print('The number of labeled nodes = %d\n'%len(lb_idx))
  print('The number of neighbors in kNN graph = %d\n'%param.k)
  print('The initial kernel parameter heuristics = %s\n'%param.sigma)
  print('Predction error rate with the inital graph  = %.3f\n'%err_rate0)
  print('Predction error rate with the optimized graph = %.3f\n'%err_rate)

  IPython.embed()


def testAEWAS (N=20):

  ## Create swiss roll data
  npts = 600
  #prev = 0.05
  c = 1
  nloops = 1.5
  var1 = 0.1
  var2 = 1.5
  shuffle = False
  eps = np.inf
  gamma = 20
  prev = 0.20
  
  # n = 1000
  # noise_level = 3# 1 or 2
  # var1 = 0.0001
  # var2 = 0.001
  # prev = 0.25

  # Parameters for AEW
  k = 10 # The number of neighbors in kNN graph
  sigma = 'local-scaling' # Kernel parameter heuristics 'median' or 'local-scaling'
  max_iter = 100
  param = SL.MSALPParameters(k=k, sigma=sigma, max_iter=max_iter)
  # --------------------------------------------------
  sparse = False
  verbose = False

  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=prev)

  all_hits1 = []
  all_hits2 = []

  for expt in xrange(N):
    print('Experiment: %i'%(expt+1))

    # if expt == 0:
    #   X,Y = du.generate_syndata(noise_level, nc=2, prev=prev, var1=var1, var2=var2, display=True)
    # else:
    #   X,Y = du.generate_syndata(noise_level, nc=2, prev=prev, var1=var1, var2=var2, display=False)
    # Y = np.squeeze(Y[:,0])  
    
    X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var1, var2=var2, shuffle=shuffle)
    # if expt == 0:
    #   pos_inds = (Y==1).nonzero()[0]
    #   neg_inds = (Y==0).nonzero()[0]

    #   plt.scatter(X[pos_inds,0], X[pos_inds,1], color='b', label='pos')
    #   plt.scatter(X[neg_inds,0], X[neg_inds,1], color='r', label='neg')

    #   plt.legend()
    #   plt.show()

    # k = 10
    # sigma = 'local-scaling'
    # # A = createEpsilonGraph (X, eps=eps, gamma=gamma)
    # # A1,sigma1 = du.generate_nngraph (X, k=k, sigma=sigma)

    # # Perform AEW 
    # max_iter = 100
    # param = SL.MSALPParameters(k=k, sigma=sigma, max_iter=max_iter)
    A2,A1 = SL.AEW(X,param)

    # ## Initialize naiveAS
    # pi = prev

    np_init = 1
    nn_init = 1
    n_init = np_init + nn_init
    initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
    initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
    init_labels = {p:1 for p in initp_pt}
    for p in initn_pt: init_labels[p] = 0

    lAS1 = ASI.naiveAS (prms)
    lAS1.initialize(A1, init_labels)
    lAS2 = ASI.naiveAS (prms)
    lAS2.initialize(A2, init_labels)

    # plotData(X, None, lAS1.f, lAS1.labels, fid=0)
    # plotData(X, None, lAS2.f, lAS2.labels, fid=1)

    hits1 = [n_init]
    hits2 = [n_init]
    K = 150
    for i in xrange(K):

      if verbose:
        print('Iter %i out of %i'%(i+1,K))
      idx1 = lAS1.getNextMessage()
      idx2 = lAS2.getNextMessage()
      lAS1.setLabelCurrent(Y[idx1])
      lAS2.setLabelCurrent(Y[idx2])
      hits1.append(hits1[-1]+Y[idx1])
      hits2.append(hits2[-1]+Y[idx2])

      # plotData(X, None, lAS1.f, lAS1.labels, fid=0)
      # plotData(X, None, lAS2.f, lAS2.labels, fid=1)

      if verbose:
        print('')
    print ('Hits1: %i\t Hits2: %i\n'%(hits1[-1], hits2[-1]))
    all_hits1.append(hits1)
    all_hits2.append(hits2)

  all_hits1 = np.array(all_hits1)
  all_hits2 = np.array(all_hits2)

  m1 = np.mean(all_hits1, axis=0)
  m2 = np.mean(all_hits2, axis=0)
  s1 = np.std(all_hits1, axis=0)
  s2 = np.std(all_hits2, axis=0)

  means = {'NN':m1, 'AEW':m2}
  sds = {'NN':s1, 'AEW':s2}
  import plotnips as pltnips
  pltnips.plot_expts (means, sds, keys=None, prev=prev, save='aew%.2f'%prev)
  # pltnips.plot_expts (means, sds, keys=None, prev=prev, title='NN vs AEW')

  import IPython
  IPython.embed()
  


def testMultipleKernelAS ():

  ## Create swiss roll data
  npts = 600
  prev = 0.05
  c = 1
  nloops = 1.5
  var = 0.2
  shuffle = False
  eps = 2
  gamma = 10
  
  X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)

  k = 10
  sigma = 'local-scaling'
  A1,sigma1 = du.generate_nngraph (X, k=k, sigma=sigma)
  A2 = createEpsilonGraph (X, eps=eps, gamma=gamma)

  ## Initialize naiveAS
  pi = prev
  sparse = False
  verbose = True

  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi)

  np_init = 1
  nn_init = 1
  n_init = np_init + nn_init
  initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
  initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
  init_labels = {p:1 for p in initp_pt}
  for p in initn_pt: init_labels[p] = 0

  K = 200
  nB = 2 # number of bandits/experts
  gamma = 0.1
  beta = 0.0 # leave this as 0
  exp3params = AAS.EXP3Parameters (gamma = gamma, T=K, nB=nB, beta=beta)
  rwmparams = AAS.RWMParameters (gamma = gamma, T=K, nB=nB)

  mAS1 = AAS.EXP3NaiveAS (prms, exp3params)
  mAS1.initialize([A1,A2], init_labels)
  mAS2 = AAS.RWMNaiveAS (prms, rwmparams)
  mAS2.initialize([A1,A2], init_labels)

  # plotData(X, None, mAS1.f, mAS1.labels, fid=0)
  plotData(X, None, mAS2.f, mAS2.ASexperts[0].labels, fid=1)

  hits1 = [n_init]
  hits2 = [n_init]
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx1 = mAS1.getNextMessage()
    idx2 = mAS2.getNextMessage()
    print idx1, idx2
    mAS1.setLabelCurrent(Y[idx1])
    mAS2.setLabelCurrent(Y[idx2])
    hits1.append(hits1[-1]+Y[idx1])
    hits2.append(hits2[-1]+Y[idx2])

    # plotData(X, None, mAS1.f, mAS1.labels, fid=0)
    plotData(X, None, mAS2.f, mAS2.ASexperts[0].labels, fid=1)

    print('')

  IPython.embed()


if __name__ == '__main__':
  # testSwissRolls()
  # testNaiveAS()
  # testWNAS()
  # testRWNAS()
  # testMPCKAS()
  # testNPKAS()
  # testAEW()
  testAEWAS ()
  # testMultipleKernelAS ()
  # pass
