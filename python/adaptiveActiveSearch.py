from __future__ import division, print_function
import time
import itertools
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

import activeSearchInterface as ASI
import similarityLearning as SL

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_sqrt (W):
	# Given PSD W, Finds PSD Q such that W = Q*Q.
	S,U = nlg.eigh(W) 
	return U.dot(np.diag(np.sqrt(S))).dot(U.T)

class adaptiveKernelAS (ASI.genericAS):

	def __init__ (self, W0, T, ASparams=ASI.Parameters(), SLparams = SL.SPSDParameters()):

		self.ASparams = ASparams
		self.kAS = None

		self.SLparams = SLparams
		self.spsdSL = SL.SPSD(SLparams) # will learn this when given data.

		self.W = W0
		self.sqrtW = matrix_sqrt(self.W)
		self.T = T

		self.epoch_itr = 0
		self.itr = 0

		self.initialized = False
		self.start_point = None
		self.Xf = None

	def initialize(self, Xf, init_labels = {}):
		# Reset self.kAS
		if self.Xf is None:
			self.Xf = Xf
		if not np.allclose (self.sqrtW, np.eye(self.W.shape[0])):
			# import IPython
			# IPython.embed()
			Xf = ss.csr_matrix(self.sqrtW).dot(Xf)

		self.kAS = ASI.kernelAS(self.ASparams)
		self.kAS.initialize (Xf, init_labels)

		if not self.initialized or self.start_point is None:
			self.start_point = self.kAS.start_point
			self.initialized = True

	def relearnSimilarity (self, params=None):

		init_labels = {i:self.kAS.labels[i] for i in self.kAS.labeled_idxs}
		L = [(self.Xf[:,i], l) for i,l in init_labels.items()]

		print("Running SPSD for relearning similarity.")

		self.spsdSL.initialize(L,self.W,params)
		self.spsdSL.runSPSD()
		
		print("Finished learning new similarity.")
		
		self.W = self.spsdSL.getW()
		self.sqrtW = self.spsdSL.getSqrtW()

		print("Reinitializing Active Search.")

		self.initialize(self.Xf, init_labels)

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

		# PERFORM RELEARNING
		if self.itr > self.T:
			self.relearnSimilarity()
			self.itr = 0
			self.epoch_itr += 1

	def getStartPoint(self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	def resetLabel (self, idx, lbl):
		if self.kAS is None:
			raise Exception ("Has not been initialized.")
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