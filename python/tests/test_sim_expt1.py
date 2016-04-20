from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio
import scipy.sparse as ss, scipy.sparse.linalg as sslg
# import matplotlib.pyplot as plt

from multiprocessing import Pool

import time
import os, os.path as osp
import csv
import cPickle as pick


import adaptiveActiveSearch as AAS
import activeSearchInterface as ASI
import similarityLearning as SL

import IPython

np.set_printoptions(suppress=True, precision=5, linewidth=100)

# data_dir = osp.join(os.getenv('HOME'),  'Research/Data/ActiveSearch/Kyle/data/KernelAS')
# results_dir = osp.join(os.getenv('HOME'),  'Classes/10-725/project/ActiveSearch/results')
data_dir = os.getenv('AS_DATA_DIR')
results_dir = os.getenv('AS_RESULTS_DIR')

def load_covertype (sparse=True, fname=None):

	if fname is None:
		fname = osp.join(data_dir, 'covtype.data')
	fn = open(fname,'r')
	data = csv.reader(fn)

	r = 54

	classes = []
	if sparse:
		Y = []
		rows = []
		cols = []
		sdat = []

		c = 0
		for line in data:
			y = int(float(line[-1]))
			Y.append(y)
			if y not in classes: classes.append(y)

			xvec = np.array(line[:r]).astype(float)
			xcol = xvec.nonzero()[0].tolist()

			rows.extend(xcol)
			cols.extend([c]*len(xcol))
			sdat.extend(xvec[xcol].tolist())

			c += 1

		X = ss.csr_matrix((sdat, (rows, cols)), shape=(r, c))

	else:

		X = []
		Y = []
		for line in data:
			X.append(np.asarray(line[:54]).astype(float))
			y = int(float(line[-1]))
			Y.append(y)
			if y not in classes: classes.append(y)

		X = np.asarray(X).T

	fn.close()
	Y = np.asarray(Y)
	return X, Y, classes

def load_higgs (sparse=True, fname = None):

	if fname is None:
		fname = osp.join(data_dir, 'HIGGS.csv')
	else:
		if fname[-3:] == '.cpk':
			with open(fname,'r') as fh: X,Y = pick.load(fh) 
	fn = open(fname,'r')
	data = csv.reader(fn)

	r = 28

	classes = []
	if sparse:
		Y = []
		rows = []
		cols = []
		sdat = []

		c = 0
		for line in data:
			y = int(float(line[0]))
			Y.append(y)
			if y not in classes: classes.append(y)

			xvec = np.array(line[1:]).astype(float)
			xcol = xvec.nonzero()[0].tolist()

			rows.extend(xcol)
			cols.extend([c]*len(xcol))
			sdat.extend(xvec[xcol].tolist())

			c += 1

		X = ss.csr_matrix((sdat, (rows, cols)), shape=(r, c))

	else:

		X = []
		Y = []
		for line in data:
			X.append(np.asarray(line[1:]).astype(float))
			y = int(float(line[0]))
			Y.append(y)
			if y not in classes: classes.append(y)

		X = np.asarray(X).T

	fn.close()

	Y = np.asarray(Y)
	return X, Y, classes


def stratified_sample (X, Y, classes, strat_frac=0.1):

	inds = []
	for c in classes:
		c_inds = (Y==c).nonzero()[0]
		c_num = int(len(c_inds)*strat_frac)
		inds.extend(c_inds[nr.permutation(len(c_inds))[:c_num]].tolist())

	Xs = X[:,inds]
	Ys = Y[inds]
	return Xs, Ys


def change_prev (X,Y,prev=0.05):
	# Changes the prevalence of positves to 0.05
	pos = Y.nonzero()[0]
	neg = (Y==0).nonzero()[0]

	npos = len(pos)
	nneg = len(neg)
	npos_prev = prev*nneg/(1-prev)#int(round(npos*prev))

	prev_idxs = pos[nr.permutation(npos)[:npos_prev]].tolist() + neg.tolist()
	nr.shuffle(prev_idxs)
	
	return X[:,prev_idxs], Y[prev_idxs]
	

def return_average_positive_neighbors (X, Y, k, use_for=True, seed=0):
	Y = np.asarray(Y)
	
	nr.seed(seed)
	n_sample_pos = 200;
	pos_inds = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_sample_pos,replace=False)]
	npos = len(pos_inds)

	if use_for:
		MsimY = []
		ii = npos
		for pind in pos_inds:
			print (ii)
			ii -= 1
			posM = np.array(X[:,pind].T.dot(X).todense()).squeeze()
			posM[pind] = -np.inf
			MsimInds = posM.argsort()[-k:]
			MsimY.append(sum(Y[MsimInds]))
		return sum(MsimY)/(npos*k)

	else:
		Xpos = X[:,pos_inds]
		posM = np.array(Xpos.T.dot(X).todense())
		posM[xrange(npos), pos_inds] = -np.inf
		MsimInds = posM.argsort(axis=1)[:,-k:]

		MsimY = Y[MsimInds]

		return MsimY.sum(axis=None)/(npos*k)

def test_covtype (seed=0):
	nr.seed(seed)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 999
	T = 1000

	sl_alpha = 0.01
	sl_C1 = 0.0
	sl_C2 = 1.0
	sl_gamma = 0.01
	sl_margin = 0.01
	sl_sampleR = 5000
	sl_epochs = 30
	sl_npairs_per_epoch = 30000
	sl_nneg_per_pair = 1
	sl_batch_size = 1000
	
	strat_frac = 0.2
	X0,Y0,classes = load_covertype(sparse=sparse)
	if strat_frac >= 1.0:
		X, Y = X0, Y0
	else:
		X, Y = stratified_sample(X0, Y0, classes, strat_frac=strat_frac)

	d,n = X.shape

	X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
	X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization

	cl = 4
	Y = (Y==cl)

	n_pos = 300
	n_neg = 30000

	n_samples_pos = np.min([n_pos,len(Y.nonzero()[0])])
	n_samples_neg = np.min([n_neg,len((Y == 0).nonzero()[0])])

	W0 = np.eye(d)
	print 'Loaded the data'

	init_pt_pos = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_samples_pos,replace=False)]
	print 'Sampled the positive data'

	init_pt_neg = (Y == 0).nonzero()[0][nr.choice(len((Y == 0).nonzero()[0]),n_samples_neg,replace=False)]
	print 'Sampled Negative Data'

	idxs = np.concatenate((init_pt_pos, init_pt_neg))
	X_sampled = X[:,idxs]
	Y_sampled = Y[idxs]

	W0 = np.eye(d)

	slprms = SL.SPSDParameters(alpha=sl_alpha, C1=sl_C1, C2=sl_C2, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair, batch_size=sl_batch_size)
	# Now learn the similarity using the sampled data
	sl = SL.SPSD()
	sl.initialize(X_sampled,Y_sampled,W0,slprms)
	sl.runSPSD()
	W = sl.getW()
	sqrtW = sl.getSqrtW()

	X1 = ss.csr_matrix(sqrtW).dot(X)
	knn = 100

	# Run Active Search
	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)

	kAS = ASI.linearizedAS (prms)
	aAS = AAS.adaptiveLinearizedAS(W0, T, prms, slprms)

	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),10,replace=False)]

	kAS.initialize(X, init_labels={p:1 for p in init_pt})
	aAS.initialize(X1, init_labels={p:1 for p in init_pt})

	hits1 = [10]
	hits2 = [10]

	for i in xrange(K):

		idx1 = kAS.getNextMessage()
		idx2 = aAS.getNextMessage()

		kAS.setLabelCurrent(Y[idx1])
		aAS.setLabelCurrent(Y[idx2])
		print('')

		hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])
	
	# Calculate KNN Accuracy
	knn_avg_native = return_average_positive_neighbors(X, Y, knn, seed)
	knn_avg_learned = return_average_positive_neighbors(X1, Y, knn, seed)

	save_results = {'kAS': hits1, 'aAS':hits2, 'knn_native':knn_avg_native, 'knn_learned':knn_avg_learned}
	fname = 'covertype/expt1_seed_%d_npos_%d_nneg_%d.cpk'%(seed, n_samples_pos, n_samples_neg)
	fname = osp.join(results_dir, fname)

	with open(fname, 'w') as fh: pick.dump(save_results, fh)


def test_higgs (seed=0):
	nr.seed(seed)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 999
	T = 1000

	sl_alpha = 0.01
	sl_C1 = 0.0
	sl_C2 = 1.0
	sl_gamma = 0.01
	sl_margin = 0.01
	sl_sampleR = 5000
	sl_epochs = 30
	sl_npairs_per_epoch = 30000
	sl_nneg_per_pair = 1
	sl_batch_size = 1000
	
	# Stratified sampling
	strat_frac = 0.02
	t1 = time.time()
	X,Y,classes = load_higgs(sparse=sparse)
	print ('Time taken to load %.2fs'%(time.time()-t1))
	if strat_frac < 1.0:
		X, Y = stratified_sample(X, Y, classes, strat_frac=strat_frac)

	# Changing prevalence of +
	prev = 0.025
	X,Y = change_prev (X,Y,prev=prev)
	d,n = X.shape

	X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
	X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization

	n_pos = 100
	n_neg = 10000

	n_samples_pos = np.min([n_pos,len(Y.nonzero()[0])])
	n_samples_neg = np.min([n_neg,len((Y == 0).nonzero()[0])])

	W0 = np.eye(d)
	print 'Loaded the data'

	init_pt_pos = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_samples_pos,replace=False)]
	print 'Sampled the positive data'

	init_pt_neg = (Y == 0).nonzero()[0][nr.choice(len((Y == 0).nonzero()[0]),n_samples_neg,replace=False)]
	print 'Sampled Negative Data'

	idxs = np.concatenate((init_pt_pos, init_pt_neg))
	X_sampled = X[:,idxs]
	Y_sampled = Y[idxs]

	W0 = np.eye(d)

	slprms = SL.SPSDParameters(alpha=sl_alpha, C1=sl_C1, C2=sl_C2, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair, batch_size=sl_batch_size)
	# Now learn the similarity using the sampled data
	sl = SL.SPSD()
	sl.initialize(X_sampled,Y_sampled,W0,slprms)
	sl.runSPSD()
	W = sl.getW()
	sqrtW = sl.getSqrtW()

	X1 = ss.csr_matrix(sqrtW).dot(X)
	knn = 100

	# Run Active Search
	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)

	kAS = ASI.linearizedAS (prms)
	aAS = AAS.adaptiveLinearizedAS(W0, T, prms, slprms)

	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),2,replace=False)]

	kAS.initialize(X, init_labels={p:1 for p in init_pt})
	aAS.initialize(X1, init_labels={p:1 for p in init_pt})

	hits1 = [2]
	hits2 = [2]

	for i in xrange(K):

		idx1 = kAS.getNextMessage()
		idx2 = aAS.getNextMessage()

		kAS.setLabelCurrent(Y[idx1])
		aAS.setLabelCurrent(Y[idx2])
		print('')

		hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])
	
	# Calculate KNN Accuracy
	knn_avg_native = return_average_positive_neighbors(X, Y, knn, seed)
	knn_avg_learned = return_average_positive_neighbors(X1, Y, knn, seed)

	save_results = {'kAS': hits1, 'aAS':hits2, 'knn_native':knn_avg_native, 'knn_learned':knn_avg_learned}
	fname = 'higgs/expt1_seed_%d_npos_%d_nneg_%d.cpk'%(seed, n_samples_pos, n_samples_neg)
	fname = osp.join(results_dir, fname)

	with open(fname, 'w') as fh: pick.dump(save_results, fh)


if __name__ == '__main__':
	import sys
	dset = 1
	num_expts = 3
	if len(sys.argv) > 1:
		try:
			dset = int(sys.argv[1])
		except:
			dset = 1
		if dset not in [1,2]:
			dset = 1
	if len(sys.argv) > 2:
		try:
			num_expts = int(sys.argv[2])
		except:
			num_expts = 3
		if num_expts > 10:
			num_expts = 10
		elif num_expts < 1:
			num_expts = 1

	test_funcs = {1:test_covtype, 2:test_higgs}

	seeds = nr.choice(10^6,num_expts,replace=False)

	if num_expts == 1:
		print ('Running 1 experiment')
		#test_funcs[dset](seeds[0])
		test_funcs[dset](109)
	else:
		print ('Running %i experiments'%num_expts)
		pl = Pool(num_expts)
		pl.map(test_funcs[dset], seeds)
