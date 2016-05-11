
from __future__ import division, print_function
import time
import itertools
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.sparse.linalg as ssl
import scipy.spatial.distance as ssd, scipy.linalg as slg
import cvxopt as cvx

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))


### ----------------------------------------------------------------------- ###
# Based on:
# Self-supervised online metric learning with low 
# rank constraint for scene categorization
# -- http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6508918

class SPSDParameters:
	# Parameters for SPSD
	def __init__(self, alpha=1, C1=0, C2=1, gamma=1, margin=None, 
			sampleR=-1, batch_size=100,
			epochs=4, npairs_per_epoch = 100000, nneg_per_pair = 4, 
			verbose=True, sparse=False, sqrt_eps=1e-6):

		self.alpha = alpha
		self.C1 = C1
		self.C2 = C2
		self.epochs = epochs
		self.gamma = gamma
		self.margin = 1 if margin is None else margin
		self.batch_size = batch_size

		self.sampleR = sampleR

		
		self.npairs_per_epoch = npairs_per_epoch
		self.nneg_per_pair = nneg_per_pair

		self.verbose = verbose
		self.sparse = sparse
		self.sqrt_eps = sqrt_eps

	def copy(self):
		return SPSDParameters(self.alpha, self.C1, self.C2, self.gamma, self.margin, 
			self.sampleR, self.batch_size,
			self.epochs, self.npairs_per_epoch, self.nneg_per_pair, 
			self.verbose, self.sparse, self.sqrt_eps)

class SPSD:

	# Class for Stochastic Proximal Subgradient Descent for bi-linear sim. learning
	def __init__(self, params=SPSDParameters()):

		self.params = params		
		self.W = None
		self.sqrtW = None

	def initialize (self, X, Y, W0, params = None):
		# Can also be used to reset the instance
		# X: r x n --> r is number of features, n is number of data points
		if params is not None:
			self.params = params
		self.X = np.array(X.todense()) # assuming that it isn't too large
		self.Y = np.array(Y)
		self.Pinds = self.Y.nonzero()[0]
		self.Ninds = (self.Y==0).nonzero()[0]
		
		self.W0 = W0

		self.W = None
		self.sqrtW = None

		self.has_learned = False

	def prox (self, M, l=None):
		# Evaluate the prox operator for nuclear norm projected onto PSD matrices
		if l is None: l = self.params.gamma

		S,U = nlg.eigh((M+M.T)/2)
		Sp = np.where(S>l,S-l,0)

		return U.dot(np.diag(Sp)).dot(U.T)

	def evalL(self, W, r, margin=None):
		# Evalute an element of the loss at 
		if margin is None: margin = self.params.margin
		return np.max([0, margin - r[0].T.dot(W).dot(r[1]) + r[0].T.dot(W).dot(r[2])])

	def subgradG(self, W, r, nR=None, C1=None, C2=None, margin=None):
	# def subgradG(self, W, r, nR=None, C=None, margin=None):
		# Evaluate the sub-gradient of smooth part of loss function:
		# 		g(W,r) = 1/(2|R|) ||W-W_0||^2_F + C*l(W,r)
		# where l(W,(x1,x2,x3)) = max{0, 1-x1^T*W*x2 + x1^T*W*x3}
		# We get: dg(W,r) = 1/|R| (W-W_0) +C dl (W,r)
		if margin is None: margin = self.params.margin
		if C1 is None: C1 = self.params.C1
		if C2 is None: C2 = self.params.C2

		x1,x2,x3 = [np.atleast_2d(x).T for x in r]
		dl = 0 if (self.evalL(W,r,margin) <= 0) else x1.dot(x3.T-x2.T)

		return C1*(W - self.W0) + C2*dl


	# New version using epochs and more intelligent sampling
	def runSPSD (self):
		# Run the SPSD
		# if not isinstance(self.params.alpha,list):
		# 	alphas = [self.params.alpha]*self.nR
		# else: alphas = self.params.alpha
		npos = len(self.Pinds)
		nneg = len(self.Ninds)

		if npos < 2 or nneg < 1:
			# Our iterations won't work with this:
			# We need at least 2 positives and 1 negative
			self.has_learned = False # just to stay safe
			self.W = self.W0
			return

		npe = self.params.npairs_per_epoch
		npp = self.params.nneg_per_pair

		alpha = self.params.alpha
		W = self.W0
		itr = 0
		# Loop through the data multiple times
		# Each time, sample positive pairs
		# For each positive pair, sample negative points to form triplets
		print ('Starting to learn the similarity')
		# print ('Number of Positive pairs per Epoc: ', npe)
		self.W_prev = W

		print ('C1: %f\nC2: %.3f\nnumPOS: %i\nnumNEG: %i'%(self.params.C1, self.params.C2, npos, nneg))
		
		for epoch in xrange(self.params.epochs):
			print ('Epoch No. ', epoch+1, '   Starting...')
			pos_pair_inds = [pind for pind in itertools.permutations(xrange(npos),2)]
			nr.shuffle(pos_pair_inds)
			nBatch = 0
			subGrad = 0
			for pi1,pi2 in pos_pair_inds[:npe]:
				for ni in nr.permutation(nneg)[:npp]:
					r = (self.X[:,self.Pinds[pi1]], self.X[:,self.Pinds[pi2]], self.X[:,self.Ninds[ni]])
					nBatch += 1
					subGrad += self.subgradG(W,r)
					if(nBatch >= self.params.batch_size):
						W = self.prox(W - alpha*subGrad, l = alpha*self.params.gamma)
						nBatch = 0
						subGrad = 0
						# change = W - self.W_prev
						# change_frob = nlg.norm(change, ord='fro')
						# print ('Difference Norm: ', change_frob)
						# print ('Difference Initial: ', nlg.norm(W - self.W0, ord='fro'))
						self.W_prev = W
						#W = self.prox(W - alpha*self.subgradG(W,r), l = alpha*self.params.gamma)
						itr += 1
			if(nBatch > 0):
				W = self.prox(W - alpha*subGrad, l = alpha*self.params.gamma)
				# change = W - self.W_prev
				# change_frob = nlg.norm(change, ord='fro')
				# print ('Difference Norm: ', change_frob)
				# print ('Difference Initial: ', nlg.norm(W - self.W0, ord='fro'))
				self.W_prev = W
				#W = self.prox(W - alpha*self.subgradG(W,r), l = alpha*self.params.gamma)
				itr += 1
		
		self.has_learned = True
		self.W = W

	def check_has_learned (self):
		return self.has_learned

	def getW (self):
		# Get matrix W after SPGD
		if self.W is None:
			raise Exception ("SPSD has not been run yet.")
		return self.W

	def getSqrtW (self):
		# Get Q such that W = Q*Q
		if self.W is None:
			raise Exception ("SPSD has not been run yet.")
		if self.sqrtW is None:
			S,U = nlg.eigh(self.W)
			if not np.allclose(self.W, self.W.T) or np.any(S < - self.params.sqrt_eps):
				raise Exception ("Matrix Squareroot did not get PSD matrix.")
			S = np.where (np.abs(S) < self.params.sqrt_eps, 0, S)
			self.sqrtW = U.dot(np.diag(np.sqrt(S))).dot(U.T)
		return self.sqrtW

### ----------------------------------------------------------------------- ###
### ----------------------------------------------------------------------- ###
# Based on:
# Integrating Constraints and Metric Learning in Semi-Supervised Clustering
# -- http://www.cs.utexas.edu/~ml/papers/semi-icml-04.pdf

# We are really worried about only a single cluster so making appropriate changes
# Since this is different from the paper, I will explain the changes.
# Step 1: Initialize single cluster for positives
# Step 2: Repeat until convergence:
#   Step 2a: Assign clusters for unlabeled nodes
# 	Step 2b: Estimate mean
#	Step 2c: Update metric for single cluster

class MPCKParameters:
	# Parameters for SPSD
	def __init__(self, max_iters=100, metric_thresh=1e-3, mu_thresh=1e-3, 
					sqrt_eps=1e-6, verbose=True):

		self.max_iters = max_iters
		self.metric_thresh = metric_thresh
		self.mu_thresh = mu_thresh
		self.sqrt_eps = sqrt_eps

		self.verbose = verbose

	def copy(self):
		return MPCKParameters(	self.max_iters, self.metric_thresh, 
								self.mu_thresh, self.sqrt_eps, self.verbose)

class MPCK(object):

	def __init__(self, params=MPCKParameters()):

		self.params = params

	def initialize(self, Xf, labels, A0 = None, pi=None):
		self.Xf = Xf
		self.r, self.n = Xf.shape

		self.labels = labels
		self.Pinds = (labels==1).nonzero()[0]
		self.Ninds = (labels==0).nonzero()[0]

		self.pi = 0.5 if pi is None else pi

		# number of positive entries to label
		self.Lpos = int(self.n*self.pi - len(self.Pinds))
		self.npos = self.Lpos + len(self.Pinds)

		self.sqrtA = None
		self.A = np.eye(self.r) if A0 is None else A0

		self.has_learned = False


	def computeMean (self, inds):
		return np.mean(self.Xf[:, inds], axis=1)

	def initializeClusters (self):
		# Step 1: Initialize the mean from the constraints
		self.muP = self.computeMean(self.Pinds)

	def assignClusters (self):
		# Step 2a: Initialize the mean from the constraints
		# Step 2b: compute mean
		unlabeledInds = (self.labels==-1).nonzero()[0]

		Xm = self.Xf.T-self.muP
		DA = np.diag(Xm.dot(self.A).dot(Xm.T))

		sortedInds = np.argsort(-DA[unlabeledInds])
		self.UPinds = unlabeledInds[sortedInds[:self.Lpos]]
		self.UNinds = unlabeledInds[sortedInds[self.Lpos:]]

		muP = self.computeMean(self.UPinds)
		close = np.allclose(muP, self.muP, atol=self.params.mu_thresh)
		self.muP = muP

		return close

	def updateMetrics (self):
		# Step 2c: update metric
		Xm = self.Xf[:,self.Pinds.tolist() + self.UPinds.tolist()].T - self.muP
		A = self.npos*nlg.inv(Xm.T.dot(Xm))
		close = np.allclose(A, self.A, atol=self.params.metric_thresh)
		self.A = A

		return close

	def runMPCK (self):

		# initialize clusters
		if self.params.verbose:
			print('Initializing clusters.')
		self.initializeClusters()

		if self.params.verbose:
			print('Running MPCK.')

		max_itr_reached = True
		for itr in xrange(self.params.max_iters):

			mu_close = self.assignClusters ()
			metric_close = self.updateMetrics()

			if mu_close and metric_close:
				max_itr_reached = False
				if self.params.verbose:
					print('Metric and Mu converged.')
				break

		if max_itr_reached and self.params.verbose:
			print('Maximum iterations reached.')

		self.has_learned = True

	def check_has_learned (self):
		return self.has_learned

	def getA (self):
		return self.A

	def getSqrtA (self):
		# Get Q such that A = Q*Q
		if self.sqrtA is None:
			S,U = nlg.eigh(self.A)
			if not np.allclose(self.A, self.A.T) or np.any(S < - self.params.sqrt_eps):
				raise Exception ("Matrix Squareroot did not get PSD matrix.")
			S = np.where (np.abs(S) < self.params.sqrt_eps, 0, S)
			self.sqrtA = U.dot(np.diag(np.sqrt(S))).dot(U.T)
		return self.sqrtA

################################################
## Based off of: papers.stevenhoi.com/ICML07NPK.pdf

class NPKParameters (object):

	def __init__ (self, c=1, delta=0.1):
		self.c = c
		self.delta = delta

class NPK (object):

	def __init__ (self, params=NPKParameters()):
		
		self.params = params

	def initialize(self, S, labels):
		self.n = S.shape[0]
		self.S = S
		self.S[xrange(self.n), xrange(self.n)] = 0
		self.D = np.squeeze(self.S.sum(1))

		self.labels = labels
		self.Pinds = (labels==1).nonzero()[0]
		self.Ninds = (labels==0).nonzero()[0]
		self.npos = len(self.Pinds)
		self.nneg = len(self.Ninds)

		L1 = np.atleast_2d(labels==1).astype(int)
		L2 = np.atleast_2d(labels==0).astype(int)

		self.T = L1.T.dot(L1) - L1.T.dot(L2) - L2.T.dot(L1)
		lti = np.tril_indices(self.n)
		self.T[lti[0], lti[1]] = 0

		self.Z = None

		self.has_learned = False

	def computeLaplacian (self):
		Dinv = np.diag(np.sqrt(1./self.D))

		self.L = (1+self.params.delta)*np.eye(self.n) - Dinv.dot(self.S).dot(Dinv)

	def solvePrimal (self):
		# Variable = Zv, Z, eps

		self.computeLaplacian()

		ncnts = int(self.npos*(self.npos-1)/2) + int(self.npos*self.nneg)
		c1 = self.L.reshape((self.n**2,1), order='F').tolist()
		c2 = self.params.c*np.ones((ncnts,1)).tolist()
		c = cvx.matrix(c1+c2)
		
		Tv = np.squeeze(self.T.reshape((self.n**2,1), order='F'))
		Tv_nz = Tv.nonzero()[0]

		assert len(Tv_nz) == ncnts
		Gl1 = np.diag(Tv)[Tv_nz,:]
		Ic = np.eye(ncnts)
		Gl = cvx.matrix(-np.r_[np.c_[Gl1, Ic], np.c_[np.zeros(Gl1.shape), Ic]])
		hl = cvx.matrix(-np.r_[np.ones((ncnts,1)), np.zeros((ncnts,1))])

		Gs = [cvx.matrix(-np.c_[np.eye(self.n**2), np.zeros((self.n**2,ncnts))])]

		hs = [cvx.matrix(np.zeros((self.n**2,1)))]

		sol = solvers.sdp(c=c, Gl=Gl, Gs=Gs, hl=hl, hs=hs)

		self.Z = np.array(sol['sk'][0])

		self.has_learned = True

	def getZ (self):
		if self.Z is None:
			raise Exception('Solver has not been called yet.')
		return self.Z
	
	def check_has_learned (self):
		return self.has_learned

###############################################################################
# Manifold-based Similarity Adaptation:
# http://papers.nips.cc/paper/5001-manifold-based-similarity-adaptation-for-label-propagation.pdf
# Python implementation of the code found at:
# http://www.bic.kyoto-u.ac.jp/pathway/krsym/software/MSALP/MSALP.zip
###############################################################################

class MSALPParameters (object):

	def __init__ (self, max_iter=100, k=10, sigma=0.5):
		self.max_iter = max_iter
		self.k = k
		self.sigma = sigma

	def A
