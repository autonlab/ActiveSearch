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

# data_dir = osp.join(os.getenv('HOME'),  'Research/Data/ActiveSearch/Kyle/data/linearizedAS')
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

def load_higgs (sparse=True, fname = None, add_bias=True):

	if fname is None:
		fname = osp.join(data_dir, 'HIGGS.csv')
	else:
		if fname[-3:] == '.cpk':
			with open(fname,'r') as fh: X,Y = pick.load(fh) 
	fn = open(fname,'r')
	data = csv.reader(fn)

	r = 28
	if add_bias:
		r += 1

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

			if add_bias:
				xvec = np.array([1.0] + line[1:]).astype(float)
			else:
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
			if add_bias:
				X.append(np.asarray([1.0]+line[1:]).astype(float))
			else:
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
	

def return_average_positive_neighbors (X, Y, k, use_for=True):
	Y = np.asarray(Y)

	pos_inds = Y.nonzero()[0]
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
	K = 1999
	T = 500

	sl_alpha = 0.01
	sl_C1 = 0.0
	sl_C2 = 1.0
	sl_gamma = 0.01
	sl_margin = 0.01
	sl_sampleR = 5000
	sl_epochs = 30
	sl_npairs_per_epoch = 30000
	sl_nneg_per_pair = 5
	sl_batch_size = 1000
	
	strat_frac = 1.0
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

	W0 = np.eye(d)
	
	num_init = 10
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),num_init,replace=False)]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C1=sl_C1, C2=sl_C2, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair, batch_size=sl_batch_size)

	kAS = ASI.linearizedAS (prms)
	aAS1 = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=True)
	aAS2 = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=False)

	kAS.initialize(X,init_labels={p:1 for p in init_pt})
	aAS1.initialize(X,init_labels={p:1 for p in init_pt})
	aAS2.initialize(X,init_labels={p:1 for p in init_pt})

	hits1 = [num_init]
	hits2 = [num_init]
	hits3 = [num_init]

	for i in xrange(K):

		idx1 = kAS.getNextMessage()
		idx2 = aAS1.getNextMessage()
		idx3 = aAS2.getNextMessage()

		kAS.setLabelCurrent(Y[idx1])
		aAS1.setLabelCurrent(Y[idx2])
		aAS2.setLabelCurrent(Y[idx3])
		print('')

		hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])
		hits3.append(hits3[-1]+Y[idx3])

	save_results = {'kAS': hits1, 'aAS_all':hits2, 'aAS2_recent':hits3}
	fname = 'covertype/expt2_seed_%d.cpk'%seed
	fname = osp.join(results_dir, fname)

	with open(fname, 'w') as fh: pick.dump(save_results, fh)


def test_higgs (seed=0):
	nr.seed(seed)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 999
	T = 200

	sl_alpha = 0.01
	sl_C1 = 0.0
	sl_C2 = 1.0
	sl_gamma = 0.01
	sl_margin = 0.01
	sl_sampleR = 5000
	sl_epochs = 30
	sl_npairs_per_epoch = 30000
	sl_nneg_per_pair = 5
	sl_batch_size = 1000
	
	# Stratified sampling
	strat_frac = 0.1
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

	W0 = np.eye(d)
	
	num_init = 10
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),num_init,replace=False)]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C1=sl_C1, C2=sl_C2, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair, batch_size=sl_batch_size)

	kAS = ASI.linearizedAS (prms)
	aAS1 = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=True)
	aAS2 = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=False)

	kAS.initialize(X,init_labels={p:1 for p in init_pt})
	aAS1.initialize(X,init_labels={p:1 for p in init_pt})
	aAS2.initialize(X,init_labels={p:1 for p in init_pt})

	hits1 = [num_init]
	hits2 = [num_init]
	hits3 = [num_init]

	for i in xrange(K):
		idx1 = kAS.getNextMessage()
		idx2 = aAS1.getNextMessage()
		idx3 = aAS2.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		aAS1.setLabelCurrent(Y[idx2])
		aAS2.setLabelCurrent(Y[idx3])
		print('')
		hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])
		hits3.append(hits3[-1]+Y[idx3])

	save_results = {'kAS': hits1, 'aAS_all':hits2, 'aAS2_recent':hits3}
	fname = 'higgs/expt2_seed_%d.cpk'%seed
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

	seeds = nr.choice(1000,num_expts,replace=False)
	# seeds = range(num_expts)
	if num_expts == 1:
		print ('Running 1 experiment')
		test_funcs[dset](seeds[0])
	else:
		print ('Running %i experiments'%num_expts)
		pl = Pool(num_expts)
		pl.map(test_funcs[dset],seeds)
