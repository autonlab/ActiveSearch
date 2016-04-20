#! /usr/bin/python
from __future__ import division
import time
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.linalg as slg, scipy.spatial.distance as ssd, scipy.sparse as ss
import scipy.io as sio
import matplotlib.pyplot as plt, matplotlib.cm as cm

import IPython # debugging

import activeSearchInterface as ASI

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def polarToCartesian (r, theta):
	return r*np.array([np.cos(theta), np.sin(theta)])

def cartesianToPolas (x, y):
	return np.array([nlg.norm([x,y]), np.arctan(y,x)])

def createSwissRolls (npts = 500, prev = 0.5, c = 1.0, nloops = 1.5, var = 0.05, shuffle=False):
	# npts 		-- number of points overall
	# prev 		-- prevalence of positive class
	# c 		-- r = c*theta
	# nloops	-- number of loops of swiss roll
	# var 		-- variance of 0-mean gaussian noise along the datapoints
	# shuffle	-- shuffle points or keep them grouped as 1/0

	std = np.sqrt(var)
	n1 = int(prev*npts);
	n2 = npts-n1

	angle_range1 = np.linspace(np.pi/2, 2*nloops*np.pi, n1)
	angle_range2 = np.linspace(np.pi/2, 2*nloops*np.pi, n2)

	X = np.empty([npts,2])
	Y = np.array(n1*[1] + n2*[0])

	for i in xrange(n1):
		a = angle_range1[i]
		X[i,:] = polarToCartesian(a*c, a) + nr.randn(1,2)*std
	for i in xrange(n2):
		a = angle_range2[i]
		X[i+n1,:] = polarToCartesian(a*c, a+np.pi) + nr.randn(1,2)*std

	if shuffle:
		shuffle_inds = nr.permutation(npts)
		X = X[shuffle_inds,:]
		Y = Y[shuffle_inds]

	return X,Y

def plotData(X, Y, f=None, labels=None, thresh=None, block=False):

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
	prms = ASI.NNParameters(sparse=sparse, verbose=verbose, normalize=normalize)

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


if __name__ == '__main__':
	# testSwissRolls()
	testNaiveAS()
	# testWNAS()
