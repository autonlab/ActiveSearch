from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
# import matplotlib.pyplot as plt

from multiprocessing import Pool

import time
import os, os.path as osp
import csv
import cPickle as pick
# import sqlparse as sql

import activeSearchInterface as ASI
import competitorsInterface as CI

import data_utils as du
import graph_utils as gu

import lapsvmp as LapSVM
import anchorGraph as AG

import IPython

def test_covtype (seed=None, prev=0.05, verbose=True, save=False):
	if seed is not None:
		nr.seed(seed)

	sparse = True
	pi = 0.5
	eta = 0.5
	K = 200
	
	t1 = time.time()
	X0,Y0,classes = du.load_covertype(sparse=sparse, normalize=True)
	## DUMMY STUFF
	# n = 1000
	# r = 20
	# X0 = ss.csc_matrix(np.r_[nr.randn(int(n/2),r), (2*nr.randn(n-int(n/2),r)+2)].T)
	# print (X0.shape)
	# Y0 = np.array([1]*int(n/2) + [0]*(n-int(n/2)))
	##
	print ('Time taken to load covtype data: %.2f'%(time.time()-t1))
	t1 = time.time()
	ag_file = osp.join(du.data_dir, 'covtype_AG_kmeans.npz')
	Z,rL = AG.load_AG(ag_file)
	print ('Time taken to load covtype AG: %.2f'%(time.time()-t1))
	
	# Changing prevalence of +
	if Y0.sum()/Y0.shape[0] < prev:
		prev = Y0.sum()/Y0.shape[0]
		X,Y = X0,Y0
	else:
		t1 = time.time()
		X,Y,inds = du.change_prev (X0,Y0,prev=prev,return_inds=True)
		Z = Z[inds, :]
		print ('Time taken to change prev: %.2f'%(time.time()-t1))

	# n = 3000
	# strat_frac = n/X.shape[1]
	strat_frac = 1.0
	if strat_frac < 1.0:
		t1 = time.time()
		X, Y, strat_inds = du.stratified_sample(X, Y, classes=[0,1], strat_frac=strat_frac,return_inds=True)
		Z = Z[strat_inds, :]
		print ('Time taken to stratified sample: %.2f'%(time.time()-t1))
	d,n = X.shape
	# IPython.embed()

	# init points
	n_init = 2
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_init,replace=False)]
	init_labels = {p:1 for p in init_pt}

	t1 = time.time()
	# Kernel AS
	ASprms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta)
	kAS = ASI.kernelAS (ASprms)
	kAS.initialize(X, init_labels=init_labels)
	print ('KAS initialized.')
	
	# NN AS
	normalize = True
	NNprms = CI.NNParameters(normalize=normalize ,sparse=sparse, verbose=verbose)
	NNAS = CI.averageNNAS (NNprms)
	NNAS.initialize(X, init_labels=init_labels)
	print ('NNAS initialized.')
	
	# # lapSVM AS
	# relearnT = 1
	# LapSVMoptions = LapSVM.LapSVMOptions()
	# LapSVMoptions.gamma_I = 1
	# LapSVMoptions.gamma_A = 1e-5
	# LapSVMoptions.NN = 6
	# LapSVMoptions.KernelParam = 0.35
	# LapSVMoptions.Verbose = False ## setting this to be false
	# LapSVMoptions.UseBias = True
	# LapSVMoptions.UseHinge = True
	# LapSVMoptions.LaplacianNormalize = False
	# LapSVMoptions.NewtonLineSearch = False
	# LapSVMoptions.Cg = 1 # PCG
	# LapSVMoptions.MaxIter = 1000  # upper bound
	# LapSVMoptions.CgStopType = 1 # 'stability' early stop
	# LapSVMoptions.CgStopParam = 0.015 # tolerance: 1.5%
	# LapSVMoptions.CgStopIter = 3 # check stability every 3 iterations
	# LapSVMprms = CI.lapSVMParameters(options=LapSVMoptions, relearnT=relearnT, sparse=False, verbose=verbose)
	# LapSVMAS = CI.lapsvmAS (LapSVMprms)
	# LapSVMAS.initialize(du.matrix_squeeze(X.todense()), init_labels=init_labels)
	# print ('LapSVMAS initialized.')

	# # anchorGraph AS
	gamma = 0.01
	AGprms = CI.anchorGraphParameters(gamma=gamma, sparse=sparse, verbose=verbose)
	AGAS = CI.anchorGraphAS (AGprms)
	AGAS.initialize(Z, rL, init_labels=init_labels)	
	print ('AGAS initialized.')

	hits_K = [n_init]
	hits_NN = [n_init]
	# hits_LSVM = [n_init]
	hits_AG = [n_init]

	print ('Time taken to initialize all approaches: %.2f'%(time.time()-t1))
	print ('Beginning experiment.')

	for i in xrange(K):

		print('Iter %i out of %i'%(i+1,K))
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		hits_K.append(hits_K[-1]+Y[idx1])

		idx2 = NNAS.getNextMessage()
		NNAS.setLabelCurrent(Y[idx2])
		hits_NN.append(hits_NN[-1]+Y[idx2])

		# idx3 = LapSVMAS.getNextMessage()
		# LapSVMAS.setLabelCurrent(Y[idx3])
		# hits_LSVM.append(hits_LSVM[-1]+Y[idx3])

		idx4 = AGAS.getNextMessage()
		AGAS.setLabelCurrent(Y[idx4])
		hits_AG.append(hits_AG[-1]+Y[idx4])
		print('')
	
	# Calculate KNN Accuracy
	IPython.embed()
	# save_results = {'kAS': hits1, 'aAS':hits2, 'knn_native':knn_avg_native, 'knn_learned':knn_avg_learned}
	# fname = 'covertype/expt1_seed_%d_npos_%d_nneg_%d.cpk'%(seed, n_samples_pos, n_samples_neg)
	# fname = osp.join(results_dir, fname)

	# with open(fname, 'w') as fh: pick.dump(save_results, fh)

def test_higgs (seed=None, prev=0.05, verbose = True, save=False):
	if seed is not None:
		nr.seed(seed)

	sparse = True
	pi = 0.5
	eta = 0.5
	K = 200
	
	t1 = time.time()
	X0,Y0,classes = du.load_higgs(sparse=sparse, normalize=True)
	## DUMMY STUFF
	# n = 1000
	# r = 20
	# X0 = ss.csc_matrix(np.r_[nr.randn(int(n/2),r), (2*nr.randn(n-int(n/2),r)+2)].T)
	# print (X0.shape)
	# Y0 = np.array([1]*int(n/2) + [0]*(n-int(n/2)))
	##
	print ('Time taken to load higgs data: %.2f'%(time.time()-t1))
	t1 = time.time()
	ag_file = osp.join(du.data_dir, 'HIGGS_AG_kmeans100.npz')
	Z,rL = AG.load_AG(ag_file)
	print ('Time taken to load HIGGS AG: %.2f'%(time.time()-t1))
	
	# Changing prevalence of +
	if Y0.sum()/Y0.shape[0] < prev:
		prev = Y0.sum()/Y0.shape[0]
		X,Y = X0,Y0
	else:
		t1 = time.time()
		X,Y,inds = du.change_prev (X0,Y0,prev=prev,return_inds=True)
		Z = Z[inds, :]
		print ('Time taken to change prev: %.2f'%(time.time()-t1))

	# n = 3000
	# strat_frac = n/X.shape[1]
	strat_frac = 1.0
	if strat_frac < 1.0:
		t1 = time.time()
		X, Y, strat_inds = du.stratified_sample(X, Y, classes=[0,1], strat_frac=strat_frac,return_inds=True)
		Z = Z[strat_inds, :]
		print ('Time taken to stratified sample: %.2f'%(time.time()-t1))
	d,n = X.shape
	# IPython.embed()

	# init points
	n_init = 1
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_init,replace=False)]
	init_labels = {p:1 for p in init_pt}

	t1 = time.time()
	# Kernel AS
	ASprms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta)
	kAS = ASI.kernelAS (ASprms)
	kAS.initialize(X, init_labels=init_labels)
	print ('KAS initialized.')
	
	# NN AS
	normalize = True
	NNprms = CI.NNParameters(normalize=normalize ,sparse=sparse, verbose=verbose)
	NNAS = CI.averageNNAS (NNprms)
	NNAS.initialize(X, init_labels=init_labels)
	print ('NNAS initialized.')
	
	# # lapSVM AS
	# relearnT = 1
	# LapSVMoptions = LapSVM.LapSVMOptions()
	# LapSVMoptions.gamma_I = 1
	# LapSVMoptions.gamma_A = 1e-5
	# LapSVMoptions.NN = 6
	# LapSVMoptions.KernelParam = 0.35
	# LapSVMoptions.Verbose = False ## setting this to be false
	# LapSVMoptions.UseBias = True
	# LapSVMoptions.UseHinge = True
	# LapSVMoptions.LaplacianNormalize = False
	# LapSVMoptions.NewtonLineSearch = False
	# LapSVMoptions.Cg = 1 # PCG
	# LapSVMoptions.MaxIter = 1000  # upper bound
	# LapSVMoptions.CgStopType = 1 # 'stability' early stop
	# LapSVMoptions.CgStopParam = 0.015 # tolerance: 1.5%
	# LapSVMoptions.CgStopIter = 3 # check stability every 3 iterations
	# LapSVMprms = CI.lapSVMParameters(options=LapSVMoptions, relearnT=relearnT, sparse=False, verbose=verbose)
	# LapSVMAS = CI.lapsvmAS (LapSVMprms)
	# LapSVMAS.initialize(du.matrix_squeeze(X.todense()), init_labels=init_labels)
	# print ('LapSVMAS initialized.')

	# # anchorGraph AS
	gamma = 0.01
	AGprms = CI.anchorGraphParameters(gamma=gamma, sparse=sparse, verbose=verbose)
	AGAS = CI.anchorGraphAS (AGprms)
	AGAS.initialize(Z, rL, init_labels=init_labels)	
	print ('AGAS initialized.')

	hits_K = [n_init]
	hits_NN = [n_init]
	# hits_LSVM = [n_init]
	hits_AG = [n_init]

	print ('Time taken to initialize all approaches: %.2f'%(time.time()-t1))
	print ('Beginning experiment.')

	for i in xrange(K):

		print('Iter %i out of %i'%(i+1,K))
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		hits_K.append(hits_K[-1]+Y[idx1])

		idx2 = NNAS.getNextMessage()
		NNAS.setLabelCurrent(Y[idx2])
		hits_NN.append(hits_NN[-1]+Y[idx2])

		# idx3 = LapSVMAS.getNextMessage()
		# LapSVMAS.setLabelCurrent(Y[idx3])
		# hits_LSVM.append(hits_LSVM[-1]+Y[idx3])

		idx4 = AGAS.getNextMessage()
		AGAS.setLabelCurrent(Y[idx4])
		hits_AG.append(hits_AG[-1]+Y[idx4])
		print('')
	
	# Calculate KNN Accuracy
	
	if save:
		save_results = {'kAS': hits_K, 
						'NNAS':hits_NN, 'knn_native':knn_avg_native, 'knn_learned':knn_avg_learned}
		# fname = 'covertype/expt1_seed_%d_npos_%d_nneg_%d.cpk'%(seed, n_samples_pos, n_samples_neg)
		# fname = osp.join(results_dir, fname)

		# with open(fname, 'w') as fh: pick.dump(save_results, fh)
	else:
		IPython.embed()

if __name__ == '__main__':
	test_covtype()
	#test_higgs()