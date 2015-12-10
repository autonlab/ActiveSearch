from __future__ import division, print_function
import time
import itertools
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))

class SPSDParameters:
	# Parameters for SPSD
	def __init__(self, alpha=1, C=1, gamma=1, margin=None, sampleR=-1, 
				 epochs=1, verbose=True, sparse=False, sqrt_eps=1e-6):
		self.alpha = alpha
		self.C = C
		self.gamma = gamma
		self.margin = 1 if margin is None else margin
		self.sampleR = sampleR
		self.epochs = epochs # Currently unused -- number of times to run through same data
		self.verbose = verbose
		self.sparse = sparse
		self.sqrt_eps = sqrt_eps


class SPSD:
	# Class for Stochastic Proximal Subgradient Descent for bi-linear sim. learning
	def __init__(self, params=SPSDParameters()):

		self.params = params		
		self.W = None
		self.sqrtW = None

	def initialize (self, L, W0, params = None):
		# Can also be used to reset the instance
		if params is not None:
			self.params = params
		self.L = L
		self.W0 = W0

		self.generateR ()

		self.W = None
		self.sqrtW = None

	def generateR (self):
		# Generate set of triplets for hinge loss
		P = [np.asarray(xy[0].todense()).squeeze() for xy in self.L if xy[1]==1]
		N = [np.asarray(xy[0].todense()).squeeze() for xy in self.L if xy[1]==0]

		if self.params.sampleR == -1:
			self.R = [(p[0],p[1],n) for p in itertools.permutations(P,2) for n in N]
			nr.shuffle(self.R)
		else:
			# Naive version just for now:
			self.R = [(p[0],p[1],n) for p in itertools.permutations(P,2) for n in N]
			sample_inds = nr.permutation(len(self.R))[:self.params.sampleR]
			self.R = [self.R[i] for i in sample_inds]

		self.nR = len(self.R)
		if self.params.verbose:
			print ("Number of triplets: %i\n"%self.nR)


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

	def subgradG(self, W, r, nR=None, C=None, margin=None):
		# Evaluate the sub-gradient of smooth part of loss function:
		# 		g(W,r) = 1/(2|R|) ||W-W_0||^2_F + C*l(W,r)
		# where l(W,(x1,x2,x3)) = max{0, 1-x1^T*W*x2 + x1^T*W*x3}
		# We get: dg(W,r) = 1/|R| (W-W_0) +C dl (W,r)
		if margin is None: margin = self.params.margin
		if C is None: C = self.params.C
		if nR is None: nR = self.nR

		x1,x2,x3 = [np.atleast_2d(x).T for x in r]
		dl = 0 if (self.evalL(W,r,margin) == 0) else x1.dot(x3.T-x2.T)

		#return (1/nR) * (W-self.W0) + C*dl
		return (W-self.W0) + C*dl

	def runSPSD (self):
		# Run the SPSD
		if not isinstance(self.params.alpha,list):
			alphas = [self.params.alpha]*self.nR
		else: alphas = self.params.alpha

		W = self.W0 # or maybe I?
		itr = 0
		for r,alpha in zip(self.R,alphas):
			W = self.prox(W - alpha*self.subgradG(W,r), l = alpha*self.params.gamma)
			# if self.params.verbose:
			# 	print ("Iteration %i of SPSD.\n"%itr)
			itr += 1
		self.W = W

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