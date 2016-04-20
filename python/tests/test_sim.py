from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio
import scipy.sparse as ss, scipy.sparse.linalg as sslg
import matplotlib.pyplot as plt
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

		MsimY =	Y[MsimInds]

		return MsimY.sum(axis=None)/(npos*k)


def test_covtype ():
	seed = 1
	nr.seed(seed)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 999
	T = 200

	sl_alpha = 0.01
	sl_C1 = 1.0
	sl_C2 = 1.0
	sl_gamma = 0.01
	sl_margin = 0.01
	sl_sampleR = 5000
	sl_epochs = 10
	sl_npairs_per_epoch = 30000
	sl_nneg_per_pair = 1
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
	
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),2,replace=False)]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C1=sl_C1, C2=sl_C2, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair, batch_size=sl_batch_size)

	kAS = ASI.linearizedAS (prms)
	# aAS = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=True)
	aAS = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=False)

	kAS.initialize(X,init_labels={p:1 for p in init_pt})
	aAS.initialize(X,init_labels={p:1 for p in init_pt})

	hits1 = [2]
	hits2 = [2]

	# for i in xrange(K):

	# 	idx1 = kAS.getNextMessage()
	# 	idx2 = aAS.getNextMessage()

	# 	kAS.setLabelCurrent(Y[idx1])
	# 	aAS.setLabelCurrent(Y[idx2])
	# 	print('')

	# 	hits1.append(hits1[-1]+Y[idx1])
	# 	hits2.append(hits2[-1]+Y[idx2])

	# fname = '%s/aas_stratfrac_%.3f_K_%i_T_%i_alpha_%.3f_gamma_%.3f_epochs_%i_batchsize_%i.npy'
	# fname = fname%(results_dir, strat_frac, K, T, sl_alpha, sl_gamma, sl_epochs, sl_batch_size)


	# save_params = [seed, K, T, strat_frac, sl_alpha, sl_C, sl_gamma, sl_margin, sl_epochs, sl_npairs_per_epoch, sl_nneg_per_pair, sl_batch_size]
	# save_results = [hits1, hits2]#, knn, knn_avg_native, knn_avg_learned]
	
	# with open(fname, 'w') as fh: pick.dump({'params':save_params, 'results':save_results}, fh)

	IPython.embed()	

	itr = range(K+1)
	plt.plot(itr, hits1, color='r', label='original AS')
	plt.plot(itr, hits2, color='b', label='adaptive AS')
	plt.xlabel('iterations')
	plt.ylabel('number of hits')
	plt.title('covertype data-set')
	plt.legend(loc=4)
	plt.show()


	# IPython.embed()

def test_covtype2 ():
	seed = 0
	nr.seed(seed)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 999
	T = 200

	sl_alpha = 0.01
	sl_C1 = 1e-5
	sl_C2 = 1.0
	sl_gamma = 0.01
	sl_margin = 0.01
	sl_sampleR = 5000
	sl_epochs = 10
	sl_npairs_per_epoch = 30000
	sl_nneg_per_pair = 1
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
	
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),2,replace=False)]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C1=sl_C1, C2=sl_C2, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair, batch_size=sl_batch_size)

	kAS = ASI.linearizedAS (prms)
	aAS1 = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=True)
	aAS2 = AAS.adaptiveLinearizedAS(W0, T, prms, slprms, from_all_data=False)

	kAS.initialize(X,init_labels={p:1 for p in init_pt})
	aAS1.initialize(X,init_labels={p:1 for p in init_pt})
	aAS2.initialize(X,init_labels={p:1 for p in init_pt})

	hits1 = [2]
	hits2 = [2]
	hits3 = [2]

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

	# fname = '%s/aas_stratfrac_%.3f_K_%i_T_%i_alpha_%.3f_gamma_%.3f_epochs_%i_batchsize_%i.npy'
	# fname = fname%(results_dir, strat_frac, K, T, sl_alpha, sl_gamma, sl_epochs, sl_batch_size)


	# save_params = [seed, K, T, strat_frac, sl_alpha, sl_C, sl_gamma, sl_margin, sl_epochs, sl_npairs_per_epoch, sl_nneg_per_pair, sl_batch_size]
	# save_results = [hits1, hits2]#, knn, knn_avg_native, knn_avg_learned]
	
	# with open(fname, 'w') as fh: pick.dump({'params':save_params, 'results':save_results}, fh)

	IPython.embed()	

	itr = range(K+1)
	plt.plot(itr, hits1, color='r', label='original AS')
	plt.plot(itr, hits2, color='b', label='adaptive1 AS')
	plt.plot(itr, hits3, color='b', label='adaptive2 AS')
	plt.xlabel('iterations')
	plt.ylabel('number of hits')
	plt.title('covertype data-set')
	plt.legend(loc=4)
	plt.show()


	# IPython.embed()

if __name__ == '__main__':
	# test_covtype()
	# X,Y,C = load_higgs()
	pass
