from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
# import matplotlib.pyplot as plt
import time
import os, os.path as osp
import csv
import cPickle as pick
# import sqlparse as sql

import adaptiveActiveSearch as AAS
import activeSearchInterface as ASI
import similarityLearning as SL
import data_utils as du

import IPython

def test_higgs ():
	
	verbose = True
	sparse = False
	pi = 0.5
	eta = 0.7

	# Stratified sampling
	strat_frac = 1.0
	t1 = time.time()
	X,Y,classes = du.load_projected_higgs(sparse=sparse)
	print ('Time taken to load %.2fs'%(time.time()-t1))
	if 0.0 < strat_frac and strat_frac < 1.0:
		X, Y = du.stratified_sample(X, Y, classes, strat_frac=strat_frac)

	# Changing prevalence of +
	prev = 0.05
	X,Y = du.change_prev (X,Y,prev=prev)
	d,n = X.shape

	# X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
	# X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization

	# Run Active Search
	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)

	kAS = ASI.kernelAS (prms)

	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),1,replace=False)]

	kAS.initialize(X, init_labels={p:1 for p in init_pt})

	hits1 = [len(init_pt)]

	for i in xrange(K):

		idx1 = kAS.getNextMessage()

		kAS.setLabelCurrent(Y[idx1])

		hits1.append(hits1[-1]+Y[idx1])
	
	IPython.embed()


if __name__ == '__main__':
	
	test_higgs()