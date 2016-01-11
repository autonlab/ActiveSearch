from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib.pyplot as plt
import time
import os, os.path as osp
import csv
import cPickle as pick
import sqlparse as sql

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

def load_higgs (sparse=True, fname = None, normalize=False):

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

	if normalize:
		if sparse:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization
		else:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization
	return X, Y, classes

def load_projected_higgs (sparse=True, fname = None, normalize=False):

	if fname is None:
		fname = osp.join(data_dir, 'HIGGS_projected.csv')
	else:
		if fname[-3:] == '.cpk':
			with open(fname,'r') as fh: X,Y = pick.load(fh) 
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

	if normalize:
		if sparse:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization
		else:
			X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
			X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization
	return X, Y, classes

def project_data (X, Y, dim = 2, num_samples = 10000, remove_samples=True, save=False, save_file=None):
	"""
	Project the data onto 2 dimensions.
	"""
	if dim != 2:
		raise NotImplementedError('Projection currently only works for 2 dimensions.')

	d,n = X.shape

	train_idx = nr.permutation(n)[:num_samples]

	X_train = X[train_idx]
	Y_train = Y[train_idx]
	if remove_samples:
		X = np.delete(X,train_idx,0)
		Y = np.delete(Y,train_idx,0)
	
	# Target feature matrix
	T = np.array([Y_train,(1.0-Y_train)]).T
	L = np.linalg.inv(X_train.dot(X_train.T)).dot(X_train.dot(T))
	X2 = L.T.dot(X)
	X2 = X2-np.min(X2)
	
	if save:
		if save_file is None:
			save_file = osp.join(data_dir, 'HIGGS_projected.csv')
		X2 = X2.todense().A
		np.savetext(save_file, np.c_[Y,X2.T], delimiter=',')
	else:
		return X2, Y

def load_sql (fname):

	fn = open(fname,'r')
	sqlstrm = sql.parsestream(fn)
  	
	for line in sqlstrm:
		IPython.embed()

	fn.close()


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
