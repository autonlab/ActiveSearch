from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
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

data_dir = osp.join('/home/sibiv',  'Research/Data/ActiveSearch/Kyle/data/KernelAS')
results_dir = osp.join('/home/sibiv',  'Classes/10-725/project/ActiveSearch/results')

def load_covertype (sparse=False):

	fname = osp.join(data_dir, 'covtype.data')
	fn = open(fname)
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
			y = int(line[-1])
			Y.append(y)
			if y not in classes: classes.append(y)

			xvec = np.array(line[:54]).astype(float)
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
			y = int(line[-1])
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

def return_average_positive_neighbors (X, Y, k):
	Y = np.asarray(Y)

	pos_inds = Y.nonzero()[0]
	Xpos = X[:,pos_inds]
	npos = len(pos_inds)

	posM = np.array(Xpos.T.dot(X).todense())
	posM[xrange(npos), pos_inds] = -np.inf
	MsimInds = posM.argsort(axis=1)[:,-k-1:]

	MsimY =	Y[MsimInds]

	return MsimY.sum(axis=None)/(npos*k)

def test_covtype ():

	seed = 0
	nr.seed(seed)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 1000
	T = 1000

	sl_alpha = 0.01
	sl_C = 0.0
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
	n_samples_pos = np.min([10,len(Y.nonzero()[0])])
	n_samples_neg = np.min([1000,len((Y == 0).nonzero()[0])])

	W0 = np.eye(d)
	print 'Loaded the data'

	init_pt_pos = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_samples_pos,replace=False)]
	print 'Sampled the positive data'

	init_pt_neg = (Y == 0).nonzero()[0][nr.choice(len((Y == 0).nonzero()[0]),n_samples_neg,replace=False)]
	print 'Sampled Negative Data'

	idxs = np.concatenate((init_pt_pos, init_pt_neg))
	X_sampled = X[:,idxs]
	Y_sampled = Y[idxs]

	slprms = SL.SPSDParameters(alpha=sl_alpha, C=sl_C, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair, batch_size=sl_batch_size)

	# Now learn the similarity using the sampled data
	sl = SL.SPSD()
	sl.initialize(X_sampled,Y_sampled,W0,slprms)
	sl.runSPSD()
	W = sl.getW()
	sqrtW = sl.getSqrtW()

	X1 = ss.csr_matrix(sqrtW).dot(X)
	knn = 200
	# print ('Average Positive for Indentity: ', return_average_positive_neighbors(X, Y, knn))
	# print ('Average Positive for learned Similarity: ', return_average_positive_neighbors(X1, Y, knn))

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)

	kAS = ASI.linearizedAS (prms)
	aAS = AAS.adaptiveLinearizedAS(W0, T, prms, slprms)

	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),1,replace=False)]

	kAS.initialize(X, init_labels={p:1 for p in init_pt})
	aAS.initialize(X1, init_labels={p:1 for p in init_pt})

	hits1 = [1]
	hits2 = [1]

	for i in xrange(K):

		idx1 = kAS.getNextMessage()
		idx2 = aAS.getNextMessage()

		kAS.setLabelCurrent(Y[idx1])
		aAS.setLabelCurrent(Y[idx2])
		print('')

		hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])
	
	# knn_avg_native = return_average_positive_neighbors(X, Y, knn)
	# knn_avg_learned = return_average_positive_neighbors(X1, Y, knn)

	fname = '%s/aas_stratfrac_%.3f_K_%i_T_%i_alpha_%.3f_gamma_%.3f_epochs_%i_batchsize_%i.npy'
	fname = fname%(results_dir, strat_frac, K, T, sl_alpha, sl_gamma, sl_epochs, sl_batch_size)


	save_params = [seed, K, T, strat_frac, sl_alpha, sl_C, sl_gamma, sl_margin, sl_epochs, sl_npairs_per_epoch, sl_nneg_per_pair, sl_batch_size, n_samples_pos, n_samples_neg]
	save_results = [hits1, hits2, W]#, knn, knn_avg_native, knn_avg_learned]
	
	with open(fname, 'w') as fh: pick.dump({'params':save_params, 'results':save_results}, fh)

	IPython.embed()	

	itr = range(K+1)
	plt.plot(itr, hits1, color='r', label='old sim')
	plt.plot(itr, hits2, color='b', label='new sim')
	plt.xlabel('iterations')
	plt.ylabel('number of hits')
	plt.title('%i pos %i neg'%(n_samples_pos, n_samples_neg))
	plt.legend(loc=4)
	plt.show()



	# Things which seem to be doing well:
	# Params1:
	# sl_alpha = 0.01
	# sl_C = 0.001
	# sl_gamma = 0.01
	# sl_margin = .1
	# sl_sampleR = 5000
	# sl_epochs = 5
	# sl_npairs_per_epoch = 20000
	# sl_nneg_per_pair = 1
	# sl_batch_size = 1000
	# strat_frac = 0.05
	# ('Average Positive for Indentity: ', 0.035510948905109488)
	# ('Average Positive for learned Similarity: ', 0.040547445255474455)

	# Params2:
	# sl_alpha = 0.01
	# sl_C = 0.001
	# sl_gamma = 0.01
	# sl_margin = .01
	# sl_sampleR = 5000
	# sl_epochs = 30
	# sl_npairs_per_epoch = 30000
	# sl_nneg_per_pair = 1
	# sl_batch_size = 1000
	# ('Average Positive for Indentity: ', 0.035510948905109488)
	# ('Average Positive for learned Similarity: ', 0.043467153284671531)

	# Params3:
	# sl_alpha = 0.1
	# sl_C = 0.001
	# sl_gamma = 0.01
	# sl_margin = .01
	# sl_sampleR = 5000
	# sl_epochs = 10
	# sl_npairs_per_epoch = 30000
	# sl_nneg_per_pair = 1
	# sl_batch_size = 1000
	# strat_frac = 0.1
	# ('Average Positive for Indentity: ', 0.04372262773722628)
	# ('Average Positive for learned Similarity: ', 0.095875912408759117)

	# sl_alpha = 0.1
	# sl_C = 0.001
	# sl_gamma = 0.01
	# sl_margin = .01
	# sl_sampleR = 5000
	# sl_epochs = 10
	# sl_npairs_per_epoch = 30000
	# sl_nneg_per_pair = 1
	# sl_batch_size = 1000
	# strat_frac = 0.5
	# ('Average Positive for Indentity: ', 0.092188638018936633)
	# ('Average Positive for learned Similarity: ', 0.20732337946103424)



if __name__ == '__main__':
	test_covtype()
