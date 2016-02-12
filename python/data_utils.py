from __future__ import division, print_function
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

import IPython

np.set_printoptions(suppress=True, precision=5, linewidth=100)

# data_dir = osp.join('/home/sibiv',  'Research/Data/ActiveSearch/Kyle/data/KernelAS')
# results_dir = osp.join('/home/sibiv',  'Classes/10-725/project/ActiveSearch/results')
data_dir = os.getenv('AS_DATA_DIR')
results_dir = os.getenv('AS_RESULTS_DIR')

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))

def min_sparse(X):
	if len(X.data) == 0:
		return 0
	m = X.data.min()
	return m if X.getnnz() == X.size else min(m, 0)

## some feature transforms
def bias_normalize_ft (X, sparse=True):
	## Data -- r x n for r features and n points
	if sparse:
		n = X.shape[1]
		X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
		X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization
		return ss.vstack([X,ss.csr_matrix(np.ones((1,X.shape[1])))]).tocsc()
	else:
		X_norms = np.sqrt((X*X).sum(axis=0)).squeeze()
		X = X/X_norms # Normalization
		return np.r_[X,np.ones((1,X.shape[1]))]

def bias_square_normalize_ft (X,sparse=True):
	if sparse:
		n = X.shape[1]
		X = ss.vstack([X,X.multiply(X)])
		X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
		X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization
		return ss.vstack([X,ss.csr_matrix(np.ones((1,X.shape[1])))]).tocsc()
	else:
		X = np.r_[X,X*X]
		X_norms = np.sqrt((X*X).sum(axis=0)).squeeze()
		X = X/X_norms # Normalization
		return np.r_[X,np.ones((1,X.shape[1]))]

def bias_square_ft (X,sparse=True):
	if sparse:
		return ss.vstack([X,X.multiply(X),ss.csr_matrix(np.ones((1,X.shape[1])))]).tocsc()
	else:
		return np.r_[X,X*X,np.ones((1,X.shape[1]))]	


def load_covertype (target=4, sparse=True, normalize=True):

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
			Y.append(int(y == target))
			if y not in classes: classes.append(y)

			xvec = np.array(line[:54]).astype(float)
			xcol = xvec.nonzero()[0].tolist()

			rows.extend(xcol)
			cols.extend([c]*len(xcol))
			sdat.extend(xvec[xcol].tolist())

			c += 1

		X = ss.csc_matrix((sdat, (rows, cols)), shape=(r, c))

	else:

		X = []
		Y = []
		for line in data:
			X.append(np.asarray(line[:54]).astype(float))
			y = int(line[-1])
			Y.append(int(y == target))
			if y not in classes: classes.append(y)

		X = np.asarray(X).T

	fn.close()

	Y = np.asarray(Y)
	if normalize:
		if sparse:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],c,c)) # Normalization
		else:
			X_norms = np.sqrt((X*X).sum(axis=0)).squeeze()
			X = X/X_norms # Normalization
	return X, Y, classes

def load_higgs (sparse=True, fname=None, normalize=True):

	if fname is None:
		fname = osp.join(data_dir, 'HIGGS.csv')

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

		X = ss.csc_matrix((sdat, (rows, cols)), shape=(r, c))

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

	if normalize:
		if sparse:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],c,c)) # Normalization
		else:
			X_norms = np.sqrt((X*X).sum(axis=0)).squeeze()
			X = X/X_norms # Normalization
	return X, Y, classes


def load_SUSY (sparse=True, fname=None, normalize=False):
	### SAME AS HIGGS EXCEPT r.
	### SHOULD COMBINE INTO ONE FUNCTION REALLY
	if fname is None:
		fname = osp.join(data_dir, 'SUSY.csv')

	fn = open(fname,'r')
	data = csv.reader(fn)

	r = 18

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

		X = ss.csc_matrix((sdat, (rows, cols)), shape=(r, c))

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

	if normalize:
		if sparse:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],c,c)) # Normalization
		else:
			X_norms = np.sqrt((X*X).sum(axis=0)).squeeze()
			X = X/X_norms # Normalization
	return X, Y, classes


def load_projected_data (sparse=True, fname=None, normalize=True):

	if fname is None:
		fname = osp.join(data_dir, 'SUSY_projected.csv')

	fn = open(fname,'r')
	data = csv.reader(fn)

	r = 2

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

		X = ss.csc_matrix((sdat, (rows, cols)), shape=(r, c))

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

	if normalize:
		if sparse:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization
		else:
			X_norms = np.sqrt((X*X).sum(axis=0)).squeeze()
			X = X/X_norms # Normalization
	return X, Y, classes


def project_data (X, Y, dim = 2, num_samples = 10000, remove_samples=True, save=False, save_file=None):
	"""
	Project the data onto 2 dimensions.
	"""
	if dim != 2:
		raise NotImplementedError('Projection currently only works for 2 dimensions.')

	d,n = X.shape

	train_idx = nr.permutation(n)[:num_samples]

	X_train = X[:,train_idx]
	Y_train = Y[:,train_idx]

	if remove_samples:
		cols_to_keep = np.where(np.logical_not(np.in1d(np.arange(n), train_idx)))[0]
		X2 = X[:,cols_to_keep]
		Y2 = Y[cols_to_keep]#np.delete(Y,train_idx,0)
	
	# Target feature matrix
	T = np.array([Y_train,(1.0-Y_train)]).T

	try:
		L = nlg.inv(X_train.dot(X_train.T).todense()).dot(X_train.dot(T))
	except:
		L = nlg.pinv(X_train.todense().T).dot(T)
		# X2inv = nlg.inv(X_train.dot(X_train.T).todense()+ 0.01*np.random.random((d,d)))
		# L = X2inv.dot(X_train.dot(T))

	X2 = ss.csr_matrix(L).T.dot(X2)
	# X2 = X2.todense().A - min_sparse(X2)

	# IPython.embed()
	
	if save:
		if save_file is None:
			save_file = osp.join(data_dir, 'SUSY_projected.csv')
		# X2 = X2.todense().A
		np.savetxt(save_file, np.c_[Y2,X2.T], delimiter=',')
	else:
		return X2, Y2

def project_data2 (X,Y,NT=10000,sparse=True):

	r,n = X.shape
	train_samp = nr.permutation(n)[:NT]

	X_train = X[:,train_samp]
	if sparse:
		X_train = X_train.todense()
	Y_train = Y[train_samp]

	T = np.array([Y_train,(1.0-Y_train)]).T

	try:
		L = nlg.inv(X_train.dot(X_train.T)).dot(X_train.dot(T))
	except:
		L = nlg.pinv(X_train.T).dot(T)
	# X2inv = nlg.inv(X_train.dot(X_train.T) + random_coeff*nr.random((r,r)))
	# L = X2inv.dot(X_train.dot(T))

	return L, train_samp


def project_data3 (X,Y,NT=10000):
	# sparse only

	r,n = X.shape
	train_samp = nr.permutation(n)[:NT]

	X_train = X[:,train_samp]
	X_train = X_train.todense()
	Y_train = Y[train_samp]

	T = np.array([Y_train,(1.0-Y_train)]).T

	try:
		L = nlg.inv(X_train.dot(X_train.T)).dot(X_train.dot(T))
	except:
		L = nlg.pinv(X_train.T).dot(T)

	cols_to_keep = np.where(np.logical_not(np.in1d(np.arange(n), train_samp)))[0]
	# rem_inds = np.ones(X.shape[1]).astype(bool)
	# rem_inds[train_samp] = False

	X = (ss.csc_matrix(L).T.dot(X[:,cols_to_keep])).tocsc()
	Y = Y[cols_to_keep]

	return X, Y, matrix_squeeze(L), train_samp

def apply_proj(X,Y,L,train_samp):
	cols_to_keep = np.where(np.logical_not(np.in1d(np.arange(X.shape[1]), train_samp)))[0]
	# rem_inds = np.ones(X.shape[1]).astype(bool)
	# rem_inds[train_samp] = False

	X = (ss.csc_matrix(L).T.dot(X[:,cols_to_keep])).tocsc()
	Y = Y[cols_to_keep]
	return X, Y

def load_sql (fname):
	# dummy function
	fn = open(fname,'r')
	sqlstrm = sql.parsestream(fn)
	
	for line in sqlstrm:
		IPython.embed()

	fn.close()

def change_prev (X, Y, prev=0.005, save=False, save_file=None, return_inds=False):
	# Changes the prevalence of positves to $prev
	pos = Y.nonzero()[0]
	neg = (Y==0).nonzero()[0]

	npos = len(pos)
	nneg = len(neg)
	npos_prev = prev*nneg/(1-prev)#int(round(npos*prev))

	prev_idxs = pos[nr.permutation(npos)[:npos_prev]].tolist() + neg.tolist()
	nr.shuffle(prev_idxs)
	
	X2, Y2 = X[:,prev_idxs], Y[prev_idxs]

	if save:
		if save_file is None:
			save_file = osp.join(data_dir, 'SUSY_projected_prev%.4f.csv'%prev)
		np.savetxt(save_file, np.c_[Y2,X2.T], delimiter=',')

	if return_inds:
		return X2, Y2, prev_idxs
	else:
		return X2, Y2


def stratified_sample (X, Y, classes, strat_frac=0.1, return_inds=False):

	inds = []
	for c in classes:
		c_inds = (Y==c).nonzero()[0]
		c_num = int(len(c_inds)*strat_frac)
		inds.extend(c_inds[nr.permutation(len(c_inds))[:c_num]].tolist())

	Xs = X[:,inds]
	Ys = Y[inds]

	if return_inds:
		return Xs, Ys, inds
	else:
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

if __name__ == '__main__':
	# pass
	t1 = time.time()
	X,Y,classes = load_covertype(sparse=True, normalize=True)
	print('Time taken to load data: %.2f'%(time.time()-t1))
	# IPython.embed()
	save_file = osp.join(data_dir, 'covtype_projected.csv')
	project_data (X, Y, save=True, save_file=save_file)
