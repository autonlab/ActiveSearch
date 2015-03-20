from __future__ import division
import numpy as np, numpy.linalg as nlg

def eigenmap(A,d):
	#function [X,n_conncomp,w] = glap_eigenmap(A,d);

	A = np.array(A)
	deg = A.sum(axis=1)
	# %if(exist('type','var') && strcmp(type,'normalized'))
	# %	L = speye(size(A,1)) - diag(1./sqrt(deg))*A*diag(1./sqrt(deg));
	# %else
	# %	L = diag(deg) - A;
	# %end
	L = np.diag(deg) - A

	if(d > A.shape[1]):
		print 'Error: required dimension larger than size of A!'
		return

	print 'Constructing Eigenmaps...'
	lam,X = nlg.eig(L)
	perm = np.argsort(lam)
	lam = lam[perm]

	th_zero = 1/A.shape[0]/1e3
	b = (lam < th_zero).sum()

	w = 1/np.sqrt(lam[b:d])
	X = np.c_[X[:,perm[:b]], X[:,perm[b:d]]*w[None,:]]
	
	return X, b, w, deg