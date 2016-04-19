from __future__ import division, print_function
import time
import itertools
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))


### ----------------------------------------------------------------------- ###
# Based on:
# Self-supervised online metric learning with low 
# rank constraint for scene categorization

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

