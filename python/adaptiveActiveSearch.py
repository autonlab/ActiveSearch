from __future__ import division, print_function
import time
import itertools
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

import activeSearchInterface as ASI
import similarityLearning as SL

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_sqrt (W, sqrt_eps=1e-6):
	# Given PSD W, Finds PSD Q such that W = Q*Q.
	S,U = nlg.eigh(W) 
	if not np.allclose(W, W.T) or np.any(S < -sqrt_eps):
		raise Exception ("Matrix Squareroot did not get PSD matrix.")
	S = np.where (np.abs(S) < sqrt_eps, 0, S)
	return U.dot(np.diag(np.sqrt(S))).dot(U.T)

### ----------------------------------------------------------------------- ###
# Based on:
# Self-supervised online metric learning with low 
# rank constraint for scene categorization

class SPSDLinearizedAS (ASI.genericAS):

	def __init__ (self, W0, T, ASparams=ASI.Parameters(), SLparams = SL.SPSDParameters(), learn_sim = True, from_all_data=False):

		self.ASparams = ASparams
		self.kAS = None

		self.SLparams = SLparams
		self.spsdSL = SL.SPSD(SLparams) # will learn this when given data.

		self.W = W0
		self.sqrtW = matrix_sqrt(self.W)
		self.T = T

		self.learn_sim = learn_sim
		self.epoch_itr = 0
		self.itr = 0

		self.initialized = False
		self.start_point = None
		self.Xf = None

		self.from_all_data = from_all_data
		self.recent_labeled_idxs = []

	def initialize(self, Xf, init_labels = {}):
		# Reset self.kAS
		if self.Xf is None:
			self.Xf = Xf
		if not np.allclose (self.sqrtW, np.eye(self.W.shape[0])):
			# import IPython
			# IPython.embed()
			Xf = ss.csr_matrix(self.sqrtW).dot(Xf)

		self.kAS = ASI.linearizedAS(self.ASparams)
		self.kAS.initialize (Xf, init_labels)

		if not self.initialized or self.start_point is None:
			self.recent_labeled_idxs = init_labels.keys()
			self.start_point = self.kAS.start_point
			self.initialized = True

	def relearnSimilarity (self):

		if not self.learn_sim:
			return

		if self.from_all_data:
			X = self.Xf[:, self.kAS.labeled_idxs]
			Y = self.kAS.labels[self.kAS.labeled_idxs]
			params = None
		else:
			if self.epoch_itr <= 0:
				params = self.SLparams.copy()
				params.C1 = 0
				params.C2 = 1
			else:
				params = self.SLparams
			X = self.Xf[:, self.recent_labeled_idxs]
			Y = self.kAS.labels[self.recent_labeled_idxs]

		print("Running SPSD for relearning similarity.")

		self.spsdSL.initialize(X,Y, self.W, params)
		self.spsdSL.runSPSD()
		
		print("Finished learning new similarity.")
		
		if self.spsdSL.check_has_learned():
			self.W = self.spsdSL.getW()
			self.sqrtW = self.spsdSL.getSqrtW()

		print("Reinitializing Active Search.")

		self.initialize(self.Xf, {i:self.kAS.labels[i] for i in self.kAS.labeled_idxs})

	def firstMessage(self,idx):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.firstMessage(idx)
		if self.start_point is None:
			self.start_point = idx

	def interestingMessage(self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.interestingMessage(idx)

	def boringMessage(self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.boringMessage(idx)

	def setLabelCurrent(self, value):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.setLabel(self.kAS.next_message, value)

	def setLabel (self, idx, lbl):
		# THIS IS WHERE WE RELEARN WHEN WE NEED TO
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.itr += 1
		display_iter = self.epoch_itr * self.T + self.itr
		self.kAS.setLabel(idx, lbl, display_iter)
		if not self.from_all_data:		
			self.recent_labeled_idxs.append(idx)
		# PERFORM RELEARNING
		if self.learn_sim and self.itr >= self.T:
			self.relearnSimilarity()
			if not self.from_all_data and self.spsdSL.check_has_learned():
				self.recent_labeled_idxs = []
			self.itr = 0
			self.epoch_itr += 1

	def getStartPoint(self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	def resetLabel (self, idx, lbl):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		ret = self.kAS.labels[idx]
		if self.kAS.labels[idx] == -1:
			self.setLabel(idx, lbl)
			return ret 
		elif self.kAS.labels[idx] == lbl:
			print("Already the same value!")
			return ret
		return self.kAS.resetLabel(idx, lbl)

	def getNextMessage (self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.getNextMessage()

	def setLabelBulk (self, idxs, lbls):
		for idx,lbl in zip(idxs,lbls):
			self.setLabel(idx,lbl)

	def pickRandomLabelMessage (self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.pickRandomLabelMessage()

	def getLabel (self,idx):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.getLabel(idx)


### ----------------------------------------------------------------------- ###
## Simple approach to just weight the kernel matrix
## based on the labeled distribution

class RWParameters (ASI.Parameters):

	def __init__ (self, lw=1, cut_connections=True, sparse=True, verbose=True):
		"""
		Parameters specifically for nearest neighbours.
		"""
		ASI.Parameters.__init__ (self, sparse=sparse, verbose=verbose)
		self.lw = lw # label weight
		self.cut_connections = cut_connections

class reweightedNaiveAS (ASI.genericAS):

	def __init__ (self, params=RWParameters()):
		ASI.genericAS.__init__ (self, params)

	def initialize(self, A, init_labels = {}):
		"""
		A 			--> n x n affinity matrix of feature values for each point.
		"""
		# Save Xf and initialize some of the variables which depend on Xf
		self.A = A
		self.n = A.shape[0]


		self.labeled_idxs = init_labels.keys()
		self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

		self.labels = np.array([-1]*self.n)
		self.labels[self.labeled_idxs] = init_labels.values()

		# Initialize some parameters and constants which are needed and not yet initialized
		self.l = (1-self.params.eta)/self.params.eta
		if self.params.w0 is None:
			self.params.w0 = 1/self.n

		if self.params.cut_connections and len(self.labeled_idxs) > 0:
			pos_labels = (self.labels==1).nonzero()[0]
			neg_labels = (self.labels==0).nonzero()[0]
			if len(pos_labels) > 0 and len(neg_labels) > 0:
				self.A[np.ix_(pos_labels, neg_labels)] = 0
				self.A[np.ix_(neg_labels, pos_labels)] = 0

		if self.params.lw  > 0:
			self.Lmat = np.atleast_2d(np.where(self.labels==-1, 0, self.labels))
			AL = self.A + self.params.lw*self.Lmat.T.dot(self.Lmat)
		else:
			AL = self.A
		# Set up some of the initial values of some matrices
		B = np.where(self.labels==-1, 1/(1+self.params.w0),self.l/(1+self.l))
		D = np.squeeze(AL.sum(1)) ##
		Dinv = 1./D
		BDinv = np.diag(np.squeeze(B*Dinv))

		self.q = (1-B)*np.where(self.labels==-1,self.params.pi,self.labels) # Need to update q every iteration
		I_A = np.eye(self.n) - BDinv.dot(AL)

		self.f =  nlg.solve(I_A, self.q)
		# Setting iter/start_point
		# If batch initialization is done, then start_point is everything given
		if len(self.labeled_idxs) > 0:
			if len(self.labeled_idxs) == 0:
				self.start_point = self.labeled_idxs[0]
			else:
				self.start_point = [eid for eid in self.labeled_idxs]
			# Finding the next message to show -- get the current max element
			uidx = np.argmax(self.f[self.unlabeled_idxs])
			self.next_message = self.unlabeled_idxs[uidx]
			# Now that a new message has been selected, mark it as unseen
			self.seen_next = False 

			self.iter = 0
			self.hits = [sum(init_labels.values())]		

		if self.params.verbose:
			print ("Done with the initialization.")
		
		self.initialized = True

	def firstMessage(self, idx):
		# Assuming this is always +ve. Can be changed otherwise
		# Need to check whether this does the right thing.

		if not self.initialized:
			raise Exception ("Has not been initialized with data")

		if self.iter >= 0:
			print("First message has already been set. Treating this as a positive.")
		else:
			self.start_point = idx
		self.setLabel(idx, 1)

	def interestingMessage(self):
		if self.next_message is None:
			if self.iter < 0:
				raise Exception("The algortithm has not been initialized. There is no current message.")
			else:
				raise Exception("I don't know how you got here.")
		if not self.seen_next:
			raise Exception ("This message has not been requested/seen yet.")
		self.setLabel(self.next_message, 1)

	def boringMessage(self):
		if self.next_message is None:
			if self.iter < 0:
				raise Exception("The algortithm has not been initialized. There is no current message.")
			else:
				raise Exception("I don't know how you got here.")
		if not self.seen_next:
			raise Exception ("This message has not been requested/seen yet.")
		self.setLabel(self.next_message, 0)

	def setLabelCurrent(self, value):
		if self.next_message is None:
			if self.iter < 0:
				raise Exception("The algortithm has not been initialized. There is no current message.")
			else:
				raise Exception("I don't know how you got here.")
		if not self.seen_next:
			raise Exception ("This message has not been requested/seen yet.")
		self.setLabel(self.next_message, value)

	def setLabel (self, idx, lbl):
		# Set label for given message id

		if self.params.verbose:
			t1 = time.time()

		# just in case, force lbl to be 0 or 1
		lbl = 0 if lbl <= 0 else 1
	
		# First, some book-keeping
		# If setLabel is called without "firstMessage," then set start_point
		if self.start_point is None:
			self.start_point = idx
		self.iter += 1
		self.labels[idx] = lbl
		self.unlabeled_idxs.remove(idx)

		if self.params.cut_connections:
			if lbl == 1:
				lbl_inds = (self.labels==0).nonzero()[0]
			else:
				lbl_inds = (self.labels==1).nonzero()[0]
			self.A[idx, lbl_inds] = 0
			self.A[lbl_inds, idx] = 0

		if self.params.lw  > 0:
			self.Lmat[0, idx] = lbl
			AL = self.A + self.params.lw*self.Lmat.T.dot(self.Lmat)
		else:
			AL = self.A

		B = np.where(self.labels==-1, 1/(1+self.params.w0),self.l/(1+self.l))
		D = np.squeeze(AL.sum(1)) ##
		Dinv = 1./D
		BDinv = np.diag(np.squeeze(B*Dinv))
		self.q[idx] = lbl*1/(1+self.l)
		I_A = np.eye(self.n) - BDinv.dot(AL)

		self.f =  nlg.solve(I_A, self.q)

		# Some more book-keeping
		self.labeled_idxs.append(idx)
		if self.iter == 0:
			self.hits.append(lbl)
		else:
			self.hits.append(self.hits[-1] + lbl)

		# Finding the next message to show -- get the current max element
		uidx = np.argmax(self.f[self.unlabeled_idxs])
		self.next_message = self.unlabeled_idxs[uidx]
		# Now that a new message has been selected, mark it as unseen
		self.seen_next = False 

		if self.params.verbose:
			elapsed = time.time() - t1
			print('Iter: %i, Selected: %i, Hits: %i, Time: %f'%(self.iter, self.labeled_idxs[-1], self.hits[-1], elapsed))
			
	def getStartPoint (self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	# Need to think a bit about the math here
	# def resetLabel (self, idx, lbl):
	# 	# Reset label for given message id
	# 	# If reset label is called on something not yet set, it should do the same as setLabel

	def getNextMessage (self):
		if self.next_message is None:
			if self.iter < 0:
				raise Exception("The algortithm has not been initialized. There is no current message.")
			else:
				raise Exception("I don't know how you got here.")
		self.seen_next = True
		return self.next_message

	def setLabelBulk (self, idxs, lbls):
		for idx,lbl in zip(idxs,lbls):
			self.setLabel(idx,lbls)

	def pickRandomLabeledMessage (self):
		if iter < 0:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.labeled_idxs[nr.randint(len(self.labeled_idxs))]

	def getLabel (self, idx):
		# 0 is boring, 1 is interesting and -1 is unlabeled
		return self.labels[idx]


#####################################################################

class MPCKLinearizedAS (ASI.genericAS):

	def __init__ (self, A0, T=20, ASparams=ASI.Parameters(), MPCKparams = SL.MPCKParameters(), learn_sim = True, from_all_data=True):

		self.params = ASparams
		self.kAS = None

		self.SLparams = MPCKparams
		self.MPCKSL = SL.MPCK(self.SLparams) # will learn this when given data.

		self.A = A0
		self.T = T

		self.learn_sim = learn_sim
		self.epoch_itr = 0
		self.itr = 0

		self.initialized = False
		self.start_point = None
		self.Xf = None

		self.from_all_data = from_all_data
		self.recent_labeled_idxs = []

	def initialize(self, Xf, init_labels = {}):
		# Reset self.kAS
		if self.Xf is None:
			self.Xf = Xf
		if self.A is None:
			self.A = np.eye(self.Xf.shape[0])
		print (self.Xf.shape[0])
		self.sqrtA = matrix_sqrt(self.A)
		if not np.allclose (self.sqrtA, np.eye(self.A.shape[0])):
			# import IPython
			# IPython.embed()
			Xf = self.sqrtA.dot(Xf)

		self.kAS = ASI.linearizedAS(self.params)
		self.kAS.initialize (Xf, init_labels)

		if not self.initialized or self.start_point is None:
			self.recent_labeled_idxs = init_labels.keys()
			self.start_point = self.kAS.start_point
			self.initialized = True

	def relearnSimilarity (self):

		if not self.learn_sim:
			return

		# if self.from_all_data:
		# 	X = self.Xf[:, self.kAS.labeled_idxs]
		# 	Y = self.kAS.labels[self.kAS.labeled_idxs]
		# else:
		# 	X = self.Xf[:, self.recent_labeled_idxs]
		# 	Y = self.kAS.labels[self.recent_labeled_idxs]

		print("Running MPCK for relearning similarity.")

		self.MPCKSL.initialize(self.Xf, self.kAS.labels, self.A, self.params.pi)
		self.MPCKSL.runMPCK()
		
		print("Finished learning new similarity.")
		
		if self.MPCKSL.check_has_learned():
			self.A = self.MPCKSL.getA()
			self.sqrtA = self.MPCKSL.getSqrtA()

		print("Reinitializing Active Search.")

		self.initialize(self.Xf, {i:self.kAS.labels[i] for i in self.kAS.labeled_idxs})

	def firstMessage(self,idx):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.firstMessage(idx)
		if self.start_point is None:
			self.start_point = idx

	def interestingMessage(self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.interestingMessage(idx)

	def boringMessage(self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.boringMessage(idx)

	def setLabelCurrent(self, value):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.setLabel(self.kAS.next_message, value)

	def setLabel (self, idx, lbl):
		# THIS IS WHERE WE RELEARN WHEN WE NEED TO
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.itr += 1
		display_iter = self.epoch_itr * self.T + self.itr
		self.kAS.setLabel(idx, lbl, display_iter)
		# if not self.from_all_data:		
		# 	self.recent_labeled_idxs.append(idx)
		# PERFORM RELEARNING
		if self.learn_sim and self.itr >= self.T:
			self.relearnSimilarity()
			# if not self.from_all_data and self.spsdSL.check_has_learned():
			# 	self.recent_labeled_idxs = []
			self.itr = 0
			self.epoch_itr += 1

	def getStartPoint(self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	def resetLabel (self, idx, lbl):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		ret = self.kAS.labels[idx]
		if self.kAS.labels[idx] == -1:
			self.setLabel(idx, lbl)
			return ret 
		elif self.kAS.labels[idx] == lbl:
			print("Already the same value!")
			return ret
		return self.kAS.resetLabel(idx, lbl)

	def getNextMessage (self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.getNextMessage()

	def setLabelBulk (self, idxs, lbls):
		for idx,lbl in zip(idxs,lbls):
			self.setLabel(idx,lbl)

	def pickRandomLabelMessage (self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.pickRandomLabelMessage()

	def getLabel (self,idx):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.getLabel(idx)

################################################################
## NPK Naive AS

class NPKNaiveAS (ASI.genericAS):

	def __init__ (self, T=20, ASparams=ASI.Parameters(), NPKparams = SL.NPKParameters(), learn_sim=True):

		self.params = ASparams
		self.kAS = None

		self.SLparams = NPKparams
		self.NPKSL = SL.NPK(NPKparams) # will learn this when given data.

		self.T = T

		self.learn_sim = learn_sim
		self.epoch_itr = 0
		self.itr = 0

		self.initialized = False
		self.start_point = None
		self.A = None

		# self.from_all_data = from_all_data
		self.recent_labeled_idxs = []

	def initialize(self, A, init_labels = {}):
		# Reset self.kAS
		if self.A is None:
			self.A = A

		self.kAS = ASI.naiveAS(self.params)
		self.kAS.initialize (self.A, init_labels)

		if not self.initialized or self.start_point is None:
			self.recent_labeled_idxs = init_labels.keys()
			self.start_point = self.kAS.start_point
			self.initialized = True

	def relearnSimilarity (self):

		if not self.learn_sim:
			return
		print("Running NPKL for relearning similarity.")
		
		# if self.from_all_data:
		# 	Y = self.kAS.labels[self.kAS.labeled_idxs]
		# else:
		# 	Y = self.kAS.labels[self.recent_labeled_idxs]
		
		self.NPKSL.initialize(self.A, self.kAS.labels)
		self.NPKSL.solvePrimal() ## potentially vert slow
		
		print("Finished learning new similarity.")
		
		if self.NPKSL.check_has_learned():
			self.A = self.NPKSL.getZ()

		print("Reinitializing Active Search.")

		self.initialize(self.A, {i:self.kAS.labels[i] for i in self.kAS.labeled_idxs})

	def firstMessage(self,idx):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.firstMessage(idx)
		if self.start_point is None:
			self.start_point = idx

	def interestingMessage(self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.interestingMessage(idx)

	def boringMessage(self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.kAS.boringMessage(idx)

	def setLabelCurrent(self, value):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.setLabel(self.kAS.next_message, value)

	def setLabel (self, idx, lbl):
		# THIS IS WHERE WE RELEARN WHEN WE NEED TO
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		self.itr += 1
		display_iter = self.epoch_itr * self.T + self.itr
		self.kAS.setLabel(idx, lbl, display_iter)
		# if not self.from_all_data:		
		# 	self.recent_labeled_idxs.append(idx)
		# PERFORM RELEARNING
		if self.learn_sim and self.itr >= self.T:
			self.relearnSimilarity()
			# if not self.from_all_data and self.spsdSL.check_has_learned():
			# 	self.recent_labeled_idxs = []
			self.itr = 0
			self.epoch_itr += 1

	def getStartPoint(self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	def resetLabel (self, idx, lbl):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		ret = self.kAS.labels[idx]
		if self.kAS.labels[idx] == -1:
			self.setLabel(idx, lbl)
			return ret 
		elif self.kAS.labels[idx] == lbl:
			print("Already the same value!")
			return ret
		return self.kAS.resetLabel(idx, lbl)

	def getNextMessage (self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.getNextMessage()

	def setLabelBulk (self, idxs, lbls):
		for idx,lbl in zip(idxs,lbls):
			self.setLabel(idx,lbl)

	def pickRandomLabelMessage (self):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.pickRandomLabelMessage()

	def getLabel (self,idx):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
		return self.kAS.getLabel(idx)

################################################################
## MULTIPLE KERNELS
################################################################

def drawFromPMF (pmf):
	if sum(pmf) != 1:
		pmf = np.array(pmf)/sum(pmf)
	return np.random.choice(elements, p=list(pmf))

################################################################
## Bandit approach for multiple kernels
## Using EXP3
################################################################

class EXP3Parameters (object):

	def __init__ (self, gamma=0.1, eta=None, beta=0.0, T=None, nB=None):
		self.gamma = gamma
		if eta is None:
			if T is None:
				self.eta = gamma if eta is None else eta
			else:
				assert nB is not None
				self.eta = np.sqrt(2*np.log(nB)/(T*nB))
		else:
			self.eta = eta
		self.beta = beta 

class EXP3NaiveAS (ASI.genericAS):

	def __init__ (self, ASparams=ASI.Parameters(), EXP3params = EXP3Parameters()):

		self.params = ASparams
		self.SLparams = EXP3params

		self.ASbandits = []
		self.weights = []

		self.unlabeled_idxs = []

		self.itr = -1
		self.initialized = False

		self.next_message = None
		self.seen_next = False

	def initialize(self, As, init_labels = {}):
		
		self.As = As
		self.nbandits = len(As)
		for A in As:
			self.ASbandits.append(ASI.naiveAS(self.params))
			self.ASbandits[-1].initialize(A, init_labels)
			self.weights.append(1)

		self.weights = np.array(self.weights)
		self.pmf = self.weights/self.nbandits
		self.wsum = self.nbandits

		if len(self.init_labels) > 0:
			self.labeled_idxs = init_labels.keys()
			self.drawBandit()
			self.next_message = self.ASbandits[self.b_selected].next_message
			self.seen_next = False
			self.itr = 0

		self.initialized = True

	def firstMessage(self,idx):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		if self.itr > 0:
			print ('Not first message.')

		self.setLabel(idx, 1)

		if self.start_point is None:
			self.start_point = idx

	def interestingMessage(self):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		self.setLabelCurrent(1)

	def boringMessage(self):
		if not self.initialized:
			raise Exception ("Has not been initialized.")		
		self.setLabelCurrent(0)

	def rewardFunction (self, value):
		return value

	def lossFunction (self, value):
		return 1-value

	def computeWeights(self, bidx, pidx, value):
		if pidx in self.labeled_idxs:
			print('This node has already been labeled and computed before')
			return

		if self.SLparams.beta > 0:
			wfactor = (1/self.pmf)*self.SLparams.beta
			wfactor[bidx] += self.rewardFunction(value)/self.pmf[bidx]
			self.weights = self.weights*np.exp(wfactor)
		else:
			r = self.rewardFunction(value)/self.pmf[bidx]
			wfactor = np.exp(self.SLparams.eta*r)
			self.weight[bidx] *= wfactor

	def drawBandit (self):
		# self.computeWeights must be called before this function.
		# in every iteration, in order to account for new reward.
		bprob = self.weights/self.weights.sum()
		unif = np.ones(self.nbandits)/self.nbandits
		self.pmf = self.SLparam.gamma*unif + (1-self.SLparam.gamma)*bprob
		self.b_selected = drawFromPMF(self.pmf)

	def setLabelCurrent(self, value):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		self.setLabel(self.next_message, value)

	def setLabel (self, idx, lbl):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		if pidx in self.labeled_idxs:
			raise Exception('This node has already been labeled and computed before.')

		self.itr += 1
		lbl = 0 if lbl <= 0 else 1
		# set labels and then select next message for labeling
		for nAS in self.ASbandits:
			self.setLabel(self.next_message, lbl)
		if self.next_message == idx:
			# recompute weights only if we are using bandit's query
			self.computeWeights(self.b_selected, self.next_message, lbl)
		self.drawBandit() # don't really need to re-draw bandit
		self.next_message = self.ASbandits[self.b_selected].next_message
		self.seen_next = False

		self.labeled_idxs.append(idx)

	def getStartPoint(self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	def resetLabel (self, idx, lbl):
		# TODO: not considering this for now
		pass
		# if self.kAS is None:
		# 	raise Exception ("Has not been initialized.")
		# ret = self.kAS.labels[idx]
		# if self.kAS.labels[idx] == -1:
		# 	self.setLabel(idx, lbl)
		# 	return ret 
		# elif self.kAS.labels[idx] == lbl:
		# 	print("Already the same value!")
		# 	return ret
		# return self.kAS.resetLabel(idx, lbl)

	def getNextMessage (self):
		if self.next_message is None:
			raise Exception ("Has not been initialized.")
		return self.next_message

	def setLabelBulk (self, idxs, lbls):
		for idx,lbl in zip(idxs,lbls):
			self.setLabel(idx,lbl)

	def pickRandomLabelMessage (self):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		return self.ASbandits[0].pickRandomLabelMessage()

	def getLabel (self,idx):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		return self.ASbandits[0].getLabel(idx)


################################################################
## Randomized Weighted Majority for multiple kernels
################################################################

class RWMParameters (object):

	def __init__ (self, gamma=0.1, eta=None, T=None, nB=None):
		self.gamma = gamma
		if eta is None:
			if T is None:
				self.eta = gamma if eta is None else eta
			else:
				assert nB is not None
				self.eta = np.sqrt(2*np.log(nB)/(T*nB))
		else:
			self.eta = eta

class RWMNaiveAS (ASI.genericAS):

	def __init__ (self, ASparams=ASI.Parameters(), RWMparams = RWMParameters()):

		self.params = ASparams
		self.SLparams = RWMparams

		self.ASexperts = []
		self.weights = []

		self.unlabeled_idxs = []

		self.itr = 0
		self.initialized = False

		self.next_message = None
		self.seen_next = False

	def initialize(self, As, init_labels = {}):
		
		self.As = As
		self.nexperts = len(As)
		for A in As:
			self.ASexperts.append(ASI.naiveAS(self.params))
			self.ASexperts[-1].initialize(A, init_labels)
			self.weights.append(1)

		self.weights = np.array(self.weights)
		self.pmf = self.weights/self.nexperts

		if len(self.init_labels) > 0:
			self.labeled_idxs = init_labels.keys()
			self.computeNextMessage()

		self.initialized = True

	def firstMessage(self,idx):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		if self.itr > 0:
			print ('Not first message.')

		self.setLabel(idx, 1)

		if self.start_point is None:
			self.start_point = idx

	def interestingMessage(self):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		self.setLabelCurrent(1)

	def boringMessage(self):
		if not self.initialized:
			raise Exception ("Has not been initialized.")		
		self.setLabelCurrent(0)

	def rewardFunction (self, value):
		return value

	def lossFunction (self, value):
		return 1-value

	def rlFunction (self, value):
		return 1 if value == 1 else -1

	def computeNextMessage(self):
		# Assuming weights and such are updated.
		F = np.array([nAS.f for nAS in self.ASexperts]).T
		if self.SL.param.gamma > 0:
			unif = np.ones(self.nexperts)/self.nexperts
			ews = self.weights/self.weights.sum()
			self.pmf = self.SLparam.gamma*unif + (1-self.SLparam.gamma)*ews
		else:
			self.pmf = self.weights/self.weights.sum()

		f = F.dot(self.pmf)
		uidx = np.argmax(self.f[self.unlabeled_idxs])
		self.next_message = self.unlabeled_idxs[uidx]
		self.seen_next = False

	def computeWeights(self, pidx, value):
		if pidx in self.labeled_idxs:
			print('This node has already been labeled and computed before')
			return
		wfactor = self.rlFunction(value)*self.pmf
		self.weights = self.weights*np.exp(wfactor)

	def setLabelCurrent(self, value):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		self.setLabel(self.next_message, value)

	def setLabel (self, idx, lbl):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		if pidx in self.labeled_idxs:
			raise Exception('This node has already been labeled and computed before.')

		self.itr += 1
		lbl = 0 if lbl <= 0 else 1
		# set labels and then select next message for labeling
		for nAS in self.ASbandits:
			self.setLabel(self.next_message, lbl)
		if self.next_message == idx:
			# recompute weights only if we are able to get our reward
			self.computeWeights(self.next_message, lbl)
		self.computeNextMessage()

		self.labeled_idxs.append(idx)

	def getStartPoint(self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	def resetLabel (self, idx, lbl):
		# TODO: not considering this for now
		pass

	def getNextMessage (self):
		if self.next_message is None:
			raise Exception ("Has not been initialized.")
		return self.next_message

	def setLabelBulk (self, idxs, lbls):
		for idx,lbl in zip(idxs,lbls):
			self.setLabel(idx,lbl)

	def pickRandomLabelMessage (self):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		return self.ASexperts[0].pickRandomLabelMessage()

	def getLabel (self,idx):
		if not self.initialized:
			raise Exception ("Has not been initialized.")
		return self.ASexperts[0].getLabel(idx)


################################################################
## KTA tuning for 2 kernels
## THIS IS UNIMPLEMENTED CURRENTLY
################################################################