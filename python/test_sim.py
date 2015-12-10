from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib.pyplot as plt
import time
import os, os.path as osp
import csv

import adaptiveActiveSearch as AAS
import activeSearchInterface as ASI
import similarityLearning as SL

np.set_printoptions(suppress=True, precision=5, linewidth=100)

data_dir = osp.join(os.getenv('HOME'),  'Research/Data/ActiveSearch/Kyle/data/KernelAS')

def load_covertype (pos=4, sparse=False):

	fname = osp.join(data_dir, 'covtype.data')
	fn = open(fname)
	data = csv.reader(fn)

	r = 54

	if sparse:
		Y = []
		rows = []
		cols = []
		sdat = []

		c = 0
		for line in data:
			Y.append(int(pos==int(line[-1])))
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
			Y.append(int(pos==int(line[-1])))

		X = np.asarray(X).T

	fn.close()

	Y = np.asarray(Y)
	return X,Y

def test_covtype ():

	nr.seed(0)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 1000
	T = 20

	sl_alpha = 0.001
	sl_C = 1e-10
	sl_gamma = 1e-10
	sl_margin = 1.
	sl_sampleR = 5000

	X,Y = load_covertype(sparse=sparse)
	import IPython
	IPython.embed()

	d,n = X.shape

	W0 = np.eye(d)
	
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),2,replace=False)]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C=sl_C, gamma=sl_gamma, margin=sl_margin, sampleR=sl_sampleR)

	kAS = ASI.kernelAS (prms)
	aAS = AAS.adaptiveKernelAS(W0, T, prms, slprms)

	# kAS.initialize(X,init_labels={init_pt:1})
	aAS.initialize(X,init_labels={p:1 for p in init_pt})

	hits1 = [1]
	hits2 = [1]

	for i in xrange(K):

		# idx1 = kAS.getNextMessage()
		idx2 = aAS.getNextMessage()

		# kAS.setLabelCurrent(Y[idx1])
		aAS.setLabelCurrent(Y[idx2])

		# hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])

	import IPython
	IPython.embed()


if __name__ == '__main__':
	test_covtype()