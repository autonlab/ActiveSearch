# Written by Wei Liu (wliu@ee.columbia.edu)
# Ported to Python by Sibi Venkatesan

from __future__ import division
import time
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

import graph_utils as gu

import IPython

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))

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
		Z = np.zeros(n, m)
	Dis = sqdist(TrainData,Anchor,sparse)
	if sparse:
		val = ss.csr_matrix((n,s))
	else:
		val = np.zeros((n,s))
	pos = val.copy()
	for i in xrange(s):
		mininds = Dis.min(1)
		pos[:,i] = np.atleast_2d(mininds).T
		val[:,i] = Dis[np.arange(n),mininds].T
		Dis[np.arange(n), mindinds] = 1e+60
	# del Dis
	# ind = (pos - 1) * n + repmat(cat(arange(1,n)).T,1,s)
	if flag == 0:
		if sparse:
			raise Exception('RBF doesn\'t yet work for sparse matrices')
			sigma = np.sqrt(val[:,s]).mean()
			val = np.expm1(-val/(sigma ** 2))+1
			val = np.tile(np.atleast_2d(np.sum(val,axis=1)).T**(-1),(1,s))*(val)
	else:
		for i in xrange(n):
			xi = TrainData[:,i]
			if sparse:
				xi = xi / np.sqrt((x.multiply(x).sum())
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
	Z[cat(ind)]=cat(val)
	clear(char('val'))
	clear(char('pos'))
	clear(char('ind'))
	clear(char('TrainData'))
	clear(char('Anchor'))
	T=Z.T * Z
	rL=T - T * diag(sum_(Z) ** - 1) * T
	clear(char('T'))
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
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# % 
	# % LAE (Local Anchor Embedding)
	# % Written by Wei Liu (wliu@ee.columbia.edu)
	# % x(dX1): input data vector 
	# % U(dXs): anchor data matrix, s: the number of closest anchors 
	# % cn: the number of iterations, 5-20
	# % z: the s-dimensional coefficient vector   
	# %
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	d,s = U.shape
	z0 = np.ones(s)/s #(U'*U+1e-6*eye(s))\(U'*x); % % %
	z1 = z0.copy()
	delta = np.zeros((1,cn+2))
	delta[0] = 0
	delta[1] = 1

	beta = np.zeros((1,cn+1))
	beta[0] = 1

	for t in range(cn)
	    alpha = (delta(t)-1)/delta(t+1);
	    v = z1+alpha*(z1-z0); %% probe point
	    
	    dif = x-U*v;
	    gv =  dif'*dif/2;
	    clear dif;
	    dgv = U'*U*v-U'*x;
	    %% seek beta
	    for j = 0:100
	        b = 2^j*beta(t);
	        z = SimplexPr(v-dgv/b);
	        dif = x-U*z;
	        gz = dif'*dif/2;
	        clear dif;
	        dif = z-v;
	        gvz = gv+dgv'*dif+b*dif'*dif/2;
	        clear dif;
	        if gz <= gvz
	            beta(t+1) = b;
	            z0 = z1;
	            z1 = z;
	            break;
	        end
	    end
	    if beta(t+1) == 0
	        beta(t+1) = b;
	        z0 = z1;
	        z1 = z;
	    end
	    clear z;
	    clear dgv;
	    delta(t+2) = ( 1+sqrt(1+4*delta(t+1)^2) )/2;
	    
	    %[t,z1']
	    if sum(abs(z1-z0)) <= 1e-4
	        break;
	    end
	end
	z = z1;
	clear z0;
	clear z1;
	clear delta;
	clear beta;

