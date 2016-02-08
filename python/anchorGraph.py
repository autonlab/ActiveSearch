# Written by Wei Liu (wliu@ee.columbia.edu)
# Ported to Python by Sibi Venkatesan

from __future__ import division
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
		A = nlg.solve(matrix_squeeze(LM.todense()) + 1e-06*np.eye(m),matrix_squeeze(RM))
		A = ss.csr_matrix(A)
	else:
		A = nlg.solve(LM + 1e-06*np.eye(m),RM)

	# del LM
	# del RM
	F = Z.dot(A)
	# del Z
	if sparse:
		F1 = F.dot(ss.diags([matrix_squeeze(F.sum(0))**(-1)],[0]))
	else:
		F1 = F.dot(np.diag(np.squeeze(F.sum(0))**(-1)))

	output = F1.argmax(axis=1)+1
	# del temp
	# del F1
	# del order
	output[lbl_idxs] = lbl_vals
	return F,A,output


def AnchorGraph(TrainData, Anchor, s=5, flag=None, cn=None, sparse=True):
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
	
	Dis = sqdist(TrainData,Anchor,sparse)

	if sparse:
		val = ss.lil_matrix((n,s))
		pos = ss.lil_matrix((n,s))
	else:
		val = np.zeros((n,s))
		pos = np.zeros((n,s), dtype=int)

	for i in xrange(s):
		mininds = Dis.argmin(1)

		if sparse:
			pos[:,i] = np.atleast_2d(mininds).T
			val[:,i] = Dis[np.arange(n),mininds].T
		else:
			pos[:,i] = mininds
			val[:,i] = Dis[np.arange(n),mininds]
		Dis[np.arange(n), mininds] = 1e+60

	if sparse:
		val = val.tocsr()
		pos = pos.tocsr()
	# del Dis
	# ind = (pos - 1) * n + repmat(cat(arange(1,n)).T,1,s)
	if flag == 0:
		if sparse:
			print('Warning: Converting to dense matrix for RBF computation.')
			sigma = val[:,s-1].sqrt().mean()
			val = np.exp(-matrix_squeeze(val.todense())/(sigma ** 2))
			val = np.tile(np.atleast_2d(np.sum(val,axis=1)).T**(-1),(1,s))*(val)
			val = ss.csr_matrix(val)
		else:
			sigma = np.sqrt(val[:,s-1]).mean()
			val = np.exp(-val/(sigma ** 2))
			val = np.tile(np.atleast_2d(np.sum(val,axis=1)).T**(-1),(1,s))*(val)
	else:
		for i in xrange(n):
			xi = TrainData[:,i]
			if sparse:
				xi = xi / np.sqrt(x.multiply(x).sum())
			else:
				xi = xi / nlg.norm(xi)
				if sparse:
					U = Anchor[:,matrix_squeeze(pos[i,:].todense())]
					U = U.dot(ss.diags([(U.multiply(U)).sum(axis=0)**(-0.5)],[0]))
				else:
					U = Anchor[:,np.squeeze(pos[i,:])]
					U = U.dot(np.diag((U**2).sum(axis=0)**(-0.5)))
			val[i,:] = LAE(xi,U,cn,sparse)

		# del xi
		# del U
	for i in xrange(s):
		pinds = matrix_squeeze(pos[:,i].todense()) if sparse else pos[:,i].squeeze()
		Z[np.arange(n), pinds] = val[:,i]
	# del val
	# del pos
	# del ind
	# del TrainData
	# del Anchor

	T = Z.T.dot(Z)
	if sparse:
		rL = T - T.dot(ss.diags([Z.sum(axis=0)**(-1)],[0])).dot(T)
	else:
		rL = T - T.dot(np.diag(Z.sum(axis=0)**(-1))).dot(T)
	 # del T
	return Z,rL

def  sqdist (A, B, sparse=True):
	# % SQDIST - computes squared Euclidean distance matrix
	# %          computes a rectangular matrix of pairwise distances
	# % between points in A (given in columns) and points in B

	if sparse:
		A2 = A.multiple(A).sum(axis=0)
		B2 = A.multiple(A).sum(axis=0)
	else:
		A = matrix_squeeze(A)
		B = matrix_squeeze(B)
		A2 = np.atleast_2d((A*A).sum(axis=0))
		B2 = np.atleast_2d((B*B).sum(axis=0))
	AB = A.T.dot(B)


	if sparse:
		d = (-2*AB).tolil()
		for i in xrange(n):
			d[i,:] = d[i,:] + B2
		for i in xrange(m):
			d[:,i] = d[:,i] + A2
		return d
		# d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);
	else:
		return np.tile(A2.T, (1,B.shape[1])) + np.tile(B2, (A.shape[1],1)) - 2*AB


def LAE(x,U,cn,sparse=True):
	# % LAE (Local Anchor Embedding)
	# % x(dX1): input data vector 
	# % U(dXs): anchor data matrix, s: the number of closest anchors 
	# % cn: the number of iterations, 5-20
	# % z: the s-dimensional coefficient vector   


	d,s = U.shape
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
			z = SimplexPr(v - dgv/b)
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


if __name__ == '__main__':

	sparse = False

	dat_dir = osp.join(os.getenv('HOME'), 'opt/Anchor_Graph')
	mdat = sio.loadmat(osp.join(dat_dir, 'USPS-MATLAB-train.mat'))
	mdat_labels = sio.loadmat(osp.join(dat_dir, 'usps_label_100.mat'))
	mdat_anchors = sio.loadmat(osp.join(dat_dir, 'usps_anchor_1000.mat'))

	data = mdat['samples']
	labels = mdat['labels'].squeeze()
	label_index = mdat_labels['label_index']
	anchor = mdat_anchors['anchor'].T

	r,n = data.shape
	m = 1000
	s = 3
	cn = 10
	C = labels.max()

	# construct an AnchorGraph(m,s) with kernel weights
	# Z1, rL1 = AnchorGraph(data, anchor, s, 0, cn, sparse)
	# rate0 = np.zeros(20)
	# for i in range(20):
	# 	run_labels = {(li-1):labels[li-1] for li in label_index[i,:]}
	# 	F, A, op = AnchorGraphReg(Z1, rL1, run_labels, C, 0.01, sparse)
	# 	rate0[i] = (op!=labels).sum()/(n-len(run_labels))
	# print('\n The average classification error rate of AGR with kernel weights is %.2f.\n'%(100*np.mean(rate0)))

	# construct an AnchorGraph(m,s) with LAE weights
	Z2, rL2 = AnchorGraph(data, anchor, s, 1, cn, sparse)
	rate = np.zeros(20)
	for i in range(20):
		run_labels = {(li-1):labels[li-1] for li in label_index[i,:]}
		F, A, op = AnchorGraphReg(Z2, rL2, run_labels, C, 0.01, sparse)
		rate[i] = (op!=labels).sum()/(n-len(run_labels))

	print('\n The average classification error rate of AGR with LAE weights is %.2f.\n'%(100*np.mean(rate)))

	IPython.embed()