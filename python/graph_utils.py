## Initially written by Stefano Melacci -- 
## Converted to python by Sibi Venkatesan
from __future__ import division
import numpy as np, numpy.linalg as nlg
import scipy.sparse as ss, scipy.linalg as slg, scipy.spatial.distance as ssd

import IPython


def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))

def Adjacency (X, options):
	# 	function A = adjacency(options,X)
	# % {adjacency} computes the graph adjacency matrix.
	# %
	# %      A = adjacency(options,X)
	# %
	# %      options: a structure with the following fields
	# %               options.NN: number of nearest neighbors to use
	# %               options.GraphDistanceFunction: 'euclidean' | 'cosine'
	# %               options.GraphWeights: 'distance' | 'binary' | 'heat'
	# %               options.GraphWeightParam: width for 'heat' kernel
	# %                                         (if set to 0, it uses the mean 
	# %                                          edge length distance among
	# %                                          neighbors) 
	# %      X: N-by-D data matrix (N examples, D dimensions)
	# %
	# %      A: sparse symmetric N-by-N adjacency matrix
	# % Author: Stefano Melacci (2009)
	# %         mela@dii.unisi.it
	# %         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com


	X = np.array(X)
	n = X.shape[0]
	p = (1, options.NN+1)

	step = n if n < 500 else 500 # block size: 500

	idy = np.zeros(n*options.NN)
	DI = np.zeros(n*options.NN)
	t = 0
	s = 0

	for i1 in xrange(0, n, step):
		t = t + 1
		i2 = i1 + step
		if i2 > n: i2 = n

		Xblock = X[i1:i2,:]

		if options.GraphDistanceFunction == 'euclidean':
			dt = ssd.cdist(Xblock , X, 'euclidean')
		elif options.GraphDistanceFunction == 'cosine':
			dt = cosine(Xblock , X)
		else:
			raise Exception ('Unknown graph distance function %s.'%options.GraphDistanceFunction)

		I = np.argsort(dt,axis=1)
		Z = np.array([dt[r,:][I[r,:]] for r in range(dt.shape[0])])

		Z = Z[:,p[0]:p[1]].T # it picks the neighbors from 2nd to NN+1th
		I = I[:,p[0]:p[1]].T # it picks the indices of neighbors from 2nd to NN+1th

		g1,g2 = I.shape
		idy[s:s+g1*g2] = I.flatten('F')
		DI[s:s+g1*g2] = Z.flatten('F')
		s = s+g1*g2

	I = np.tile(range(n), (options.NN, 1))
	I = I.flatten('F')

	if options.GraphDistanceFunction == 'cosine': # only positive values
		DI = np.where(DI < 1, DI, 0) # ?

	if options.GraphWeights == 'distance' :
		A = ss.csr_matrix((DI,(I,idy)),(n,n))
	elif options.GraphWeights == 'binary':
		A = ss.csr_matrix(([1]*len(I),(I,idy)),(n,n))
	elif options.GraphWeights == 'heat':
		if options.GraphWeightParam == 0: # default (t=mean edge length)
			# % actually this computation should be
			# % made after symmetrizing the adjacecy
			# % matrix, but since it is a heuristic, this
			# % approach makes the code faster.
			t = np.mean(DI[DI!=0]) 
		else:
			t = options.GraphWeightParam

		A = ss.csr_matrix((np.exp(-DI**2/(2*t*t)),(I,idy)),(n,n))
	else:
		raise Exception('Unknown weight type')

	A = matrix_squeeze(A.todense())
	A = A + (A!=A.T).multiply(A.T) # symmetrize
	return A
	

def Laplacian (X, options):
	# function [L,options] = laplacian(options,X)  
	# % {laplacian} computes the graph Laplacian.
	# %     
	# %      [L,options] = laplacian(options,X)
	# %
	# %      options: a structure with the following fields
	# %               options.NN: number of nearest neighbors to use
	# %               options.GraphDistanceFunction: 'euclidean' | 'cosine' |
	# %               'hamming_distance'
	# %               options.GraphWeights: 'distance' | 'binary' | 'heat'
	# %               options.GraphWeightParam: width for 'heat' kernel
	# %                                         (if set to 0, it uses the mean 
	# %                                          edge length distance among
	# %                                          neighbors
	# %               options.LaplacianNormalize: 0 | 1
	# %               options.LaplacianDegree: degree of the iterated Laplacian
	# %      X: N-by-D data matrix (N examples, D dimensions)
	# %
	# %      L: sparse symmetric N-by-N Laplacian matrix
	# %      options: updated options structure with estimated heat kernel width
	# %               (only when you select to use 'heat' in the GraphWeights 
	# %                option and 'default' as GraphWeightParam)
	# %
	# % Author: Stefano Melacci (2009)
	# %         mela@dii.unisi.it
	# %         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

	W = Adjacency(X, options)
	D = matrix_squeeze(W.sum(axis=1))
	if options.LaplacianNormalize == 0:
		L = ss.diags([D],[0])-W # L = D-W
	else:
		D [D != 0] = np.sqrt(1./D [D!=0])
		D = ss.diags([D],[0])
		W = D.dot(W.dot(D))
		L = ss.eye(W.shape[0])-W  # L = I-D^-1/2*W*D^-1/2

	if options.LaplacianDegree > 1:
		L = nlg.matrix_power(L,options.LaplacianDegree)

	return L

# def euclidean(A,B):
# 	# % {euclidean} computes the Euclidean distance.
# 	# %
# 	# %      D = euclidean(A,B)
# 	# %      
# 	# %      A: M-by-P matrix of M P-dimensional vectors 
# 	# %      B: N-by-P matrix of M P-dimensional vectors
# 	# % 
# 	# %      D: M-by-N distance matrix
# 	# %
# 	# % Author: Stefano Melacci (2009)
# 	# %         mela@dii.unisi.it
# 	# %         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

# 	if A.shape[1] != B.shape[1]
# 		raise Exception('A and B must be of same dimensionality.')

# 	if A.shape[1] == 1 # if dim = 1...
# 		A = np.c_[A, np.zeros((A.shape[0],1))]
# 		B = np.c_[B, np.zeros((B.shape[0],1))]

# 	aa = np.sum(A*A, axis=1)
# 	bb = np.sum(B*B, axis=1)
# 	ab = A.dot(B.T)

# 	D = real(sqrt(repmat(aa,[1 size(bb,1)]) + repmat(bb',[size(aa,1) 1]) -2*ab));

# function D = cosine(A,B)
# 	# % {cosine} computes the cosine distance.
# 	# %
# 	# %      D = cosine(A,B)
# 	# %      
# 	# %      A: M-by-P matrix of M P-dimensional vectors 
# 	# %      B: N-by-P matrix of M P-dimensional vectors
# 	# % 
# 	# %      D: M-by-N distance matrix
# 	# %
# 	# % Author: Stefano Melacci (2009)
# 	# %         mela@dii.unisi.it
# 	# %         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

# 	if (size(A,2) ~= size(B,2))
# 		error('A and B must be of the same dimensionality.');
# 	end

# 	if (size(A,2) == 1) % if dim = 1...
# 		A = [A, zeros(size(A,1),1)];
# 		B = [B, zeros(size(B,1),1)];
# 	end

# 	aa=sum(A.*A,2);
# 	bb=sum(B.*B,2);
# 	ab=A*B';

# 	% to avoid NaN for zero norms
# 	aa((aa==0))=1; 
# 	bb((bb==0))=1;

# 	D = real(ones(size(A,1),size(B,1))-(1./sqrt(kron(aa,bb'))).*ab);
