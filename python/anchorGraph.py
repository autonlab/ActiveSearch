# Written by Wei Liu (wliu@ee.columbia.edu)
# Ported to Python by Sibi Venkatesan

from __future__ import division, print_function
import time
import os, os.path as osp

import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.sparse as ss, scipy.sparse.linalg as ssl
import scipy.linalg as slg, scipy.io as sio

import graph_utils as gu

import IPython

np.set_printoptions(suppress=True, precision=4, threshold=200)

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))

def save_AG(filename,Z,rL):
	np.savez(	filename, 
				Zdata=Z.data, Zindices=Z.indices, Zindptr=Z.indptr, Zshape=Z.shape,
				rLdata=rL.data, rLindices=rL.indices, rLindptr=rL.indptr, rLshape=rL.shape)

def load_AG(filename):
	loader = np.load(filename)
	Z = ss.csr_matrix((loader['Zdata'], loader['Zindices'], loader['Zindptr']), shape=loader['Zshape'])
	rL = ss.csr_matrix((loader['rLdata'], loader['rLindices'], loader['rLindptr']), shape=loader['rLshape'])
	return Z, rL

def sparse_argmax(X, axis=0):
	## Only does argmax over non-zero entries
	if axis == 0 and ss.isspmatrix_csc(X) is False:
		X = X.tocsc()
	elif axis == 1 and ss.isspmatrix_csr(X) is False:
		X = X.tocsr()

	group_lengths = np.diff(X.indptr)
	n_groups = len(group_lengths)
	index = np.repeat(np.arange(n_groups), group_lengths)
	maxima = np.maximum.reduceat(X.data, X.indptr[:-1])
	all_argmax = np.flatnonzero(np.repeat(maxima, group_lengths) == X.data)
	return X.indices[all_argmax[np.searchsorted(all_argmax, X.indptr[:-1])]]

def sparse_argmin(X, axis=0):
	## Only does argmin over non-zero entries
	if axis == 0 and ss.isspmatrix_csc(X) is False:
		X = X.tocsc()
	elif axis == 1 and ss.isspmatrix_csr(X) is False:
		X = X.tocsr()

	group_lengths = np.diff(X.indptr)
	n_groups = len(group_lengths)
	index = np.repeat(np.arange(n_groups), group_lengths)
	minima = np.minimum.reduceat(X.data, X.indptr[:-1])
	all_argmin = np.flatnonzero(np.repeat(minima, group_lengths) == X.data)
	return X.indices[all_argmin[np.searchsorted(all_argmin, X.indptr[:-1])]]


def SimplexPr(X):
	# % SimplexPr
	# % X(CXN): input data matrix, C: dimension, N: # samples
	# % S: the projected matrix of X onto C-dimensional simplex  
	X = np.squeeze(X)
	if len(X.shape) < 2:
		N = 1
		C = X.shape[0]
	else:
		C,N = X.shape

	if N == 1:
		T1 = -np.sort(-X)
		kk = 0
		t = T1
		for j in range(C):
			if t[j]-(np.sum(t[:j+1])-1)/(j+1) <= 0:
				kk = j
				break

		if kk == 0:
			kk = C

		theta = (np.sum(t[:kk])-1)/kk
		S = np.where(X > theta, X - theta, 0)
		# del t
		# del T1
	else:
		T1 = -np.sort(-X,axis=1)
		S = X.copy()

		for i in range(N):
			kk = 0
			t = T1[:,i]
			for j in range(C):
				if t[j]-(np.sum[t[:j+1]]-1)/(j+1) <= 0:
					kk = j
					break

			if kk == 0:
				kk = C

			theta = (np.sum(t[:kk])-1)/kk
			S[:,i] = np.where(X[:,i] > theta, X[:,i]-theta, 0)
			# del t

		# del T1
	return S


def LAE(x,U,cn,sparse=True):
	# % LAE (Local Anchor Embedding)
	# % x(dX1): input data vector 
	# % U(dXs): anchor data matrix, s: the number of closest anchors 
	# % cn: the number of iterations, 5-20
	# % z: the s-dimensional coefficient vector   

	d,s = U.shape
	if sparse:
		x = matrix_squeeze(x.todense())

	z0 = np.ones(s)/s #(U'*U+1e-6*eye(s))\(U'*x); % % %
	z1 = z0.copy()
	delta = np.zeros(cn+2)
	delta[0] = 0
	delta[1] = 1

	beta = np.zeros(cn+1)
	beta[0] = 1

	for t in range(cn):
		alpha = (delta[t]-1)/delta[t+1]
		v = z1+alpha*(z1-z0) # probe point
		
		dif = x - U.dot(v)
		gv =  dif.T.dot(dif)/2
		# del dif
		dgv = U.T.dot(U.dot(v)) - U.T.dot(x)

		# seek beta
		for j in range(101):
			b = (2**j)*beta[t]

			z = SimplexPr(matrix_squeeze(v - dgv/b))
			dif = x - U.dot(z)
			gz = dif.T.dot(dif)/2
			# del dif
			dif = z - v
			gvz = gv + dgv.T.dot(dif) + b*dif.T.dot(dif)/2
			# del dif
			if gz <= gvz:
				beta[t+1] = b
				z0 = z1
				z1 = z
				break

		if beta[t+1] == 0:
			beta[t+1] = b
			z0 = z1
			z1 = z
		# del z
		# del dgv
		delta[t+2] = (1 + np.sqrt(1+4*delta[t+1]**2))/2
		
		#[t,z1']
		if np.sum(np.abs(z1-z0)) <= 1e-4:
			break

	# del z0
	# del z1
	# del delta
	# del beta	  
	return z1


def  sqdist (A, B, sparse=True, normalized=False):
	# % SQDIST - computes squared Euclidean distance matrix
	# %          computes a rectangular matrix of pairwise distances
	# % between points in A (given in columns) and points in B

	# honestly, it doesn't make sense to have a squared distance matrix as dense.
	# If normalized, just returning the A.T*B.
	# The constants can be added later on their own
	if normalized:
		AB = A.T.dot(B)
		return - 2*AB

	if sparse:
		A2 = np.atleast_2d(np.asarray(A.multiply(A).sum(axis=0)))
		B2 = np.atleast_2d(np.asarray(B.multiply(B).sum(axis=0)))
	else:
		A = matrix_squeeze(A)
		B = matrix_squeeze(B)
		A2 = np.atleast_2d((A*A).sum(axis=0))
		B2 = np.atleast_2d((B*B).sum(axis=0))
	
	AB = A.T.dot(B)
	if sparse:
		AB = matrix_squeeze(AB.todense())


	# if sparse:
	# 	d = (-2*AB).tolil()
	# 	# for i in xrange(n):
	# 	# 	d[i,:] = d[i,:] + B2
	# 	# for i in xrange(m):
	# 	# 	d[:,i] = d[:,i] + A2.T
	# 	# return d.tocsr()
	# 	# # d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);
	# else:
	return np.tile(A2.T, (1,B.shape[1])) + np.tile(B2, (A.shape[1],1)) - 2*AB



def AnchorGraph(TrainData, Anchor, s=5, flag=1, cn=10, sparse=True, normalized=False):
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# % 
	# % AnchorGraph
	# % Written by Wei Liu (wliu@ee.columbia.edu)
	# % TrainData(dXn): input data matrix, d: dimension, n: # samples
	# % Anchor(dXm): anchor matrix, m: # anchors 
	# % s: # of closest anchors, usually set to 2-10 
	# % flag: 0 gives a Gaussian kernel-defined Z and 1 gives a LAE-optimized Z
	# % cn: # of iterations for LAE, usually set to 5-20; if flag=0, input 'cn' any value
	# % Z(nXm): output anchor-to-data regression weight matrix 
	# % rL(mXm): reduced graph Laplacian matrix
	# %
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	d,n = TrainData.shape
	m = Anchor.shape[1]

	
	if sparse:
		Z = ss.lil_matrix((n, m))
	else:
		Z = np.zeros((n, m))
	
	t1 = time.time()
	Dis = sqdist(TrainData,Anchor,sparse,normalized)
	print('Time taken to compute square distance: %.2f'%(time.time()-t1))

	if sparse:
		val = ss.lil_matrix((n,s))
		pos = ss.lil_matrix((n,s), dtype=int)
	else:
		val = np.zeros((n,s))
		pos = np.zeros((n,s), dtype=int)

	for i in xrange(s):
		print('Iteration %i out of %i for closest anchors.'%(i+1,s))
		if sparse and ss.issparse(Dis):
			mininds = sparse_argmin(Dis, 1)
		else:
			mininds = Dis.argmin(1)
		print('Found argmin.')

		if sparse:
			pos[:,i] = np.atleast_2d(mininds).T
			if normalized:
				val[:,i] = np.atleast_2d(Dis[np.arange(n),mininds]).T + 2
			else:
				val[:,i] = np.atleast_2d(Dis[np.arange(n),mininds]).T
		else:
			pos[:,i] = mininds
			if normalized:
				val[:,i] = Dis[np.arange(n),mininds] + 2
			else:
				val[:,i] = Dis[np.arange(n),mininds]
		print('Updated vals/pos.\n')

		Dis[np.arange(n), mininds] = 1e+60

	del Dis

	# del Dis
	# ind = (pos - 1) * n + repmat(cat(arange(1,n)).T,1,s)
	if flag == 0:
		if sparse: # no point in making val sparse
			pos = pos.tocsr()
			print('Warning: Converting to dense matrix for RBF computation.')
			sigma = val[:,s-1].sqrt().mean()
			val = np.exp(-matrix_squeeze(val.todense())/(sigma ** 2))
			val = np.tile(np.atleast_2d(np.sum(val,axis=1)).T**(-1),(1,s))*(val)
			# val = ss.csr_matrix(val)
		else:
			sigma = np.sqrt(val[:,s-1]).mean()
			val = np.exp(-val/(sigma ** 2))
			val = np.tile(np.atleast_2d(np.sum(val,axis=1)).T**(-1),(1,s))*(val)
	else:
		if sparse:
			pos = pos.tocsr()
			if ss.issparse(Anchor):
				Anchor = matrix_squeeze(Anchor.todense())
			for i in xrange(n):
				print('Performing LAE for %i out of %i points.'%(i+1,n),end='\r')
				# print (i)
				# t1 = time.time()
				xi = matrix_squeeze(TrainData[:,i].todense())
				if not normalized:
					xi = xi / nlg.norm(xi)
				# IPython.embed()
				U = Anchor[:,matrix_squeeze(pos[i,:].todense())]
				U = U.dot(np.diag((U**2).sum(axis=0)**(-0.5)))
				# U = U.dot(ss.diags([matrix_squeeze((U.multiply(U)).sum(axis=0))**(-0.5)],[0]))
				val[i,:] = LAE(xi,U,cn,sparse=False)
				# print ('%.5f\n'%(time.time()-t1))
			print('Performing LAE for %i out of %i points.\n'%(n,n))
			val = val.tocsr()		
		else:
			for i in xrange(n):
				xi = TrainData[:,i]
				if not normalized:
					xi = xi / nlg.norm(xi)
					
				U = Anchor[:,np.squeeze(pos[i,:])]
				U = U.dot(np.diag((U**2).sum(axis=0)**(-0.5)))
				val[i,:] = LAE(xi,U,cn,sparse)

	# del xi
	# del U
	print('Constructing Z.')
	for i in xrange(s):
		print('Iteration %i out of %i for closest anchors.'%(i+1,s))
		pinds = matrix_squeeze(pos[:,i].todense()) if sparse else pos[:,i].squeeze()
		Z[np.arange(n), pinds] = val[:,i].T
	if sparse:
		Z = Z.tocsr()
	# del val
	# del pos
	# del ind
	# del TrainData
	# del Anchor

	print('Constructing T = Z.T*Z')
	T = Z.T.dot(Z)
	print('Constructing Laplacian')
	if sparse:
		rL = T - T.dot(ss.diags([matrix_squeeze(Z.sum(axis=0))**(-1)],[0])).dot(T)
	else:
		rL = T - T.dot(np.diag(Z.sum(axis=0)**(-1))).dot(T)
	print('Done')
	 # del T
	return Z,rL


def AnchorGraphReg(Z, rL, labels, C, gamma, sparse=True, matlab_indexing=True):
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# % 
	# % AnchorGraphReg 
	# % Written by Wei Liu (wliu@ee.columbia.edu)
	# % Z(nXm): regression weight matrix
	# % rL(mXm): reduced graph Laplacian 
	# % ground(1Xn): [1 1 1 2 2 2] discrete groundtruth class labels 
	# % label_index(1Xln):  given index of labeled data
	# % gamma: regularization parameter, set to 0.001-1
	# % F(nXC): soft label scores on raw data
	# % A(mXC): soft label scores on anchors
	# % err: classification error rate on unlabeled data
	# %
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	n,m = Z.shape
	ln = len(labels)

	lbl_idxs = np.array(labels.keys())
	lbl_vals = np.array(labels.values())

	Yl = np.zeros((ln,C))

	for i in range(C):
		if matlab_indexing:
			Yl[:,i] = (lbl_vals==(i+1)).astype(int)
		else:
			Yl[:,i] = (lbl_vals==i).astype(int)
	if sparse:
		Yl = ss.csr_matrix(Yl)

	Zl = Z[lbl_idxs,:]
	LM = Zl.T.dot(Zl)+gamma*rL
	# del rL
	RM = Zl.T.dot(Yl)
	# del Yl
	# del Zl
	if sparse:
		A = ss.csr_matrix(nlg.solve(matrix_squeeze(LM.todense()) + 1e-06*np.eye(m),matrix_squeeze(RM.todense())))
	else:
		A = nlg.solve(LM + 1e-06*np.eye(m),RM)

	# del LM
	# del RM
	F = Z.dot(A)
	# del Z
	if sparse:
		F1 = F.dot(ss.diags([matrix_squeeze(F.sum(0))**(-1)],[0]))
		F1 = matrix_squeeze(F1.todense())
	else:
		F1 = F.dot(np.diag(np.squeeze(F.sum(0))**(-1)))

	output = F1.argmax(axis=1)+1
	# del temp
	# del F1
	# del order
	output[lbl_idxs] = lbl_vals
	return F,A,output