#! /usr/bin/ipython
from __future__ import division, print_function
import os, os.path as osp
import cv2
import copy
import time
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.io as sio, scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

import IPython

import activeSearchInterface as ASI
import dataUtils as du


np.set_printoptions(suppress=True, precision=5, linewidth=100)

DATA_DIR = osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/')

def two_spirals(n=100, loops=2, r=1., h=4, prev=0.5, noise_sigma=0.05,
                shuffle=False):
  """Makes two spirals of a total of n points."""

  npos = int(n * prev)
  nneg = n - npos

  pos_angles = np.linspace(0., 2 * loops * np.pi, npos).reshape(-1, 1)
  pos_range = np.linspace(0., h, npos).reshape(-1, 1)
  pos_pts = np.c_[np.cos(pos_angles), np.sin(pos_angles), pos_range]

  neg_angles = np.linspace(np.pi, (2 * loops + 1) * np.pi, nneg).reshape(-1, 1)
  neg_range = np.linspace(0., h, nneg).reshape(-1, 1)
  neg_pts = np.c_[np.cos(neg_angles), np.sin(neg_angles), neg_range]

  X = np.r_[pos_pts, neg_pts] + nr.randn(n, 3) * np.sqrt(noise_sigma)
  Y = np.array([1] * npos + [0] * nneg)

  if shuffle:
    permutation = nr.permutation(n)
    X = X[permutation]
    Y = Y[permutation]

  return X, Y


def circle_positives(n, r=1, w=0.2, g=0.1, noise_sigma=0.02, shuffle=False):
  """Creates a circle embedded in n^2 points."""
  pvals = np.linspace(-2 * r, 2 * r, n)
  X1, X2 = np.meshgrid(pvals, pvals)
  X = np.stack((X1, X2), axis=2).reshape(-1, 2)

  norm = nlg.norm(X, axis=1)
  Y = ((norm > r - w) & (norm < r + w)).astype(int)
  valid_inds = ~(((norm < r - w) & (norm > r - w - g))
                 | ((norm > r + w) & (norm < r + w + g)))
  
  X = X[valid_inds] + nr.randn(valid_inds.sum(), 2) * np.sqrt(noise_sigma)
  Y = Y[valid_inds]

  if shuffle:
    permutation = nr.permutation(n)
    X = X[permutation]
    Y = Y[permutation]

  return X, Y


def coeff():
  nr.seed(0)
  data = sio.loadmat(osp.join(DATA_DIR, 'sherry_datasets/populated_places_5000.mat'))
  A = du.matrix_squeeze(data['A'].todense())
  labels = data['labels']
  Y = (labels == 1).squeeze().astype(int)

  pi = Y.sum()/Y.shape[0]
  sparse = False
  verbose = True
  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi)
  nAS = ASI.shariAS (prms)

  n_init = 1
  init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), n_init, replace=False)]
  init_labels = {p:1 for p in init_pt}
  nAS.initialize(A, init_labels)

  K = 50
  hits = [n_init]
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx = nAS.getNextMessage()
    print(idx)
    nAS.setLabelCurrent(Y[idx])
    hits.append(hits[-1]+Y[idx])

  IPython.embed()


def test_IFAS():
  data = sio.loadmat(osp.join(DATA_DIR, 'sherry_datasets/populated_places_5000.mat'))
  # data = sio.loadmat(osp.join(DATA_DIR, 'sherry_datasets/new_nips_double.mat'))
  A = du.matrix_squeeze(data['A'].todense())
  labels = data['labels']
  Y = (labels == 1).squeeze().astype(int)
  # X = nr.randn(5, 100)**2
  # A = X.T.dot(X)
  # Y = np.array([0]*50 + [1]*50)#, nr.randint(2, size=(100,))

  pi = Y.sum()/Y.shape[0]
  sparse = False
  verbose = True
  alpha = 1 #0.00001
  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi, alpha=alpha)
  prms2 = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi, alpha=0)
  # sAS = ASI.shariAS (prms)
  imAS1 = ASI.impactFactorAS(copy.deepcopy(prms))
  imAS2 = ASI.impactFactorAS(copy.deepcopy(prms))
  imAS3 = ASI.impactFactorAS(prms2)#copy.deepcopy(prms))
  imAS1.quad_IM = False
  imAS2.quad_IM = True
  # kAS = ASI.linearizedAS(prms)

  n_init = 1
  init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), n_init, replace=False)]
  init_labels = {p:1 for p in init_pt}
  imAS1.initialize(A, init_labels)
  imAS2.initialize(A, init_labels)
  # imAS3.initialize(A, init_labels)

  # IPython.embed()

  K = 50
  hits1 = [n_init]
  hits2 = [n_init]
  hits3 = [n_init]
  for i in xrange(K):

    print('Iter %i out of %i'%(i+1,K))
    idx1 = imAS1.getNextMessage()
    idx2 = imAS2.getNextMessage()
    # idx3 = imAS3.getNextMessage()
    imAS1.setLabelCurrent(Y[idx1])
    imAS2.setLabelCurrent(Y[idx2])
    # imAS3.setLabelCurrent(Y[idx3])
    hits1.append(hits1[-1]+Y[idx1])
    hits2.append(hits2[-1]+Y[idx2])
    # hits3.append(hits3[-1]+Y[idx3])

  IPython.embed()


def test_IFAS_n(n_expts=10):
  # data = sio.loadmat(osp.join(DATA_DIR, 'sherry_datasets/populated_places_5000.mat'))
  # A = du.matrix_squeeze(data['A'].todense())
  # labels = data['labels']
  # Y = (labels == 1).squeeze().astype(int)
  
  # data = sio.loadmat(osp.join(DATA_DIR, 'sherry_datasets/wikipedia_data.mat'))
  data = sio.loadmat(osp.join(DATA_DIR, 'sherry_datasets/new_nips_double.mat'))
  A = du.matrix_squeeze(data['A'].todense())
  Y = np.zeros(A.shape[0]).astype(int)
  Y[data['positive_node_ids'].squeeze()] = 1

  verbose = True
  pi = Y.sum()/Y.shape[0]
  sparse = False
  alpha = 1
  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi, alpha=alpha)

  n = A.shape[0]
  w0 = 1 / n
  if verbose:
    M = np.diag(A.sum(0) + w0) - A
    print ("Inverting M")
  t1 = time.time()
  if verbose:
    Minv = np.matrix(nlg.inv(M))
    print("Time for inverse:", time.time() - t1)

  # IPython.embed()
  # exit()

  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi, alpha=alpha)
  prms2 = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi, alpha=0.0)
  n_init = 1

  hs1 = []
  hs2 = []
  hs3 = []

  K = 300 
  for expt in range(n_expts):
    print('\n\n EXPERIMENT %i.' % (expt + 1))
    imAS1 = ASI.impactFactorAS(copy.deepcopy(prms))
    imAS2 = ASI.impactFactorAS(copy.deepcopy(prms))
    imAS3 = ASI.impactFactorAS(copy.deepcopy(prms2))
    imAS1.quad_IM = False
    imAS2.quad_IM = True

    imAS1.Minv = Minv
    imAS2.Minv = Minv
    imAS3.Minv = Minv
    imAS1.initialize(A)
    imAS2.initialize(A)
    imAS3.initialize(A)

    init_pts = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), n_init, replace=False)]
    for init_pt in init_pts:
      imAS1.setLabel(init_pt, 1)
      imAS2.setLabel(init_pt, 1)
      imAS3.setLabel(init_pt, 1)

    hits1 = [n_init]
    hits2 = [n_init]
    hits3 = [n_init]
    for i in xrange(K):

      print('Iter %i out of %i' % (i+1,K))
      idx1 = imAS1.getNextMessage()
      imAS1.setLabelCurrent(Y[idx1])
      hits1.append(hits1[-1]+Y[idx1])

      idx2 = imAS2.getNextMessage()
      imAS2.setLabelCurrent(Y[idx2])
      hits2.append(hits2[-1]+Y[idx2])

      idx3 = imAS3.getNextMessage()      
      imAS3.setLabelCurrent(Y[idx3])
      hits3.append(hits3[-1]+Y[idx3])

    hs1.append(hits1)
    hs2.append(hits2)
    hs3.append(hits3)

  hs1 = np.array(hs1)
  hs2 = np.array(hs2)
  hs3 = np.array(hs3)

  try:
    import matplotlib.pyplot as plt

    ideal = np.arange(1, K+2)
    itr = ideal
    m1, s1 = hs1.mean(0), hs1.std(0)
    m2, s2 = hs2.mean(0), hs2.std(0)
    m3, s3 = hs3.mean(0), hs3.std(0)

    ax = plt.subplot()

    y1 = m1-s1
    y1 = np.where(y1>1, y1, 1)
    y2 = m1+s1
    y2 = np.where(y2<ideal, y2, ideal)
    ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=(1,0,0), alpha=0.2, interpolate=True)
    ax.plot(itr, y1, color=(1,0,0), linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
    ax.plot(itr, y2, color=(1,0,0), linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
    ax.plot(itr, m1, color=(1,0,0), label='Lin IM')

    y1 = m2-s2
    y1 = np.where(y1>1, y1, 1)
    y2 = m2+s2
    y2 = np.where(y2<ideal, y2, ideal)
    ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=(0,1,0), alpha=0.2, interpolate=True)
    ax.plot(itr, y1, color=(0,1,0), linewidth=1)
    ax.plot(itr, y2, color=(0,1,0), linewidth=1)
    ax.plot(itr, m2, color=(0,1,0), label='Quad IM')

    y1 = m3-s3
    y1 = np.where(y1>1, y1, 1)
    y2 = m3+s3
    y2 = np.where(y2<ideal, y2, ideal)
    ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=(0,0,1), alpha=0.2, interpolate=True)
    ax.plot(itr, y1, color=(0,0,1), linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
    ax.plot(itr, y2, color=(0,0,1), linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
    ax.plot(itr, m3, color=(0,0,1), label='No IM')

    plt.legend()
    plt.show()
  except:
    pass
  IPython.embed()


def test_CIFAR10 ():
  cifar_data = np.load(osp.join(DATA_DIR, 'LairLabData/processed_cifar_resnet.npz'))
  X_train = cifar_data['x_tra']
  Y_train = cifar_data['y_tra']
  im_train = cifar_data['im_train'].astype('uint8')
  X_test = cifar_data['x_test']
  Y_test = cifar_data['y_test']
  im_test = cifar_data['im_test'].astype('uint8')

  ims = np.r_[im_train, im_test]

  print('Loaded data.')

  cl = 9
  X = np.r_[X_train, X_test].T
  # X = ((ims[:, :1024] + ims[:, 1024:2048] + ims[:, 2048:])/3).astype(float).T
  Y = (np.r_[Y_train, Y_test].argmax(1) == cl).astype(int).squeeze()

  sparse = False
  verbose = True
  alpha = 0
  pi = Y.sum() / Y.shape[0]
  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi, alpha=alpha)

  kAS = ASI.linearizedAS(prms)

  n_init = 1
  init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), n_init, replace=False)]
  init_labels = {p:1 for p in init_pt}
  kAS.initialize(X, init_labels)

  IPython.embed()

  K = 100
  hits = [n_init]
  for i in xrange(K):
    idx = kAS.getNextMessage()
    print(idx)
    kAS.setLabelCurrent(Y[idx])
    hits.append(hits[-1]+Y[idx])

  found_imgs = ims[kAS.labeled_idxs]
  pos_imgs = []
  for img in found_imgs:
    pos_imgs.append(img.reshape((32,32,3), order='C'))

  IPython.embed()

  # for img in pos_imgs:
  #   cv2.imshow('imgs', img)
  #   cv2.waitKey(3)


def test_A9A ():
  a9a_data = np.load(osp.join(DATA_DIR, 'LairLabData/a9a/a9a_scaled_dataset.npz'))
  X_train = a9a_data['x_tra']
  Y_train = a9a_data['y_tra']
  X_test = a9a_data['x_test']
  Y_test = a9a_data['y_test']

  print('Loaded data.')

  X = np.r_[X_train, X_test].T
  Y = np.r_[Y_train, Y_test].astype(int)
  Y[Y == -1] = 0

  sparse = False
  verbose = True
  alpha = 0
  pi = Y.sum() / Y.shape[0]
  prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi, alpha=alpha)

  kAS = ASI.linearizedAS(prms)

  n_init = 1
  init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), n_init, replace=False)]
  init_labels = {p:1 for p in init_pt}
  kAS.initialize(X, init_labels)

  IPython.embed()

  K = 1000
  hits = [n_init]
  for i in xrange(K):
    idx = kAS.getNextMessage()
    print(idx)
    kAS.setLabelCurrent(Y[idx])
    hits.append(hits[-1]+Y[idx])  

  IPython.embed()


if __name__ == '__main__':
  # coeff()
  # test_IFAS()
  # test_IFAS_n(10)
  # test_CIFAR10()
  test_A9A()