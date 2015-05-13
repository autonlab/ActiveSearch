from __future__ import division
import time
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

np.set_printoptions(suppress=True, precision=5, linewidth=100)

class Parameters:

	def __init__ (self, pi=0.05, eta=0.5, w0=None, sparse=True, verbose=False):
		"""
		pi 			--> prior target probability
		eta 		--> jump probability
		w0			--> weight assigned to regularization on unlabeled points (default: 1/(#data points))
		"""
		self.pi = pi
		self.eta = eta
		self.w0 = w0
		self.sparse = sparse

		self.verbose = verbose

		alpha = 0  #not being used


class genericAS:

	def __init__ (self, params=Parameters()):
		"""
		pi 			--> prior target probability
		eta 		--> jump probability
		w0			--> 
		"""
		self.params = params

		self.start_point = None
		self.unlabeled_idxs = []
		self.labeled_idxs = []
		self.next_message = None

		self.seen_next = False

		self.hits = [] # not entirely sure what happens to this if allowed to reset
		self.labels = []

		self.iter = -1

	def saveState (self, filename):
		# save the state of the parameters and the data so that we can restart.
		raise NotImplementedError()

	def firstMessage(self, message):
		raise NotImplementedError()

	def interestingMessage(self):
		raise NotImplementedError()

	def boringMessage(self):
		raise NotImplementedError()

	def setalpha(self, alpha):
		self.params.alpha = alpha

	def setLabel (self, idx, lbl):
		raise NotImplementedError()

	def getStartPoint (self):
		return self.start_point

	def resetLabel (self, idx, lbl):
		raise NotImplementedError()

	def setLabelCurrent(self, value):
		raise NotImplementedError()

	def setLabelBulk (self, idxs, lbls):
		raise NotImplementedError()

	def getNextMessage (self):
		raise NotImplementedError()

	def pickRandomLabeledMessage (self):
		return self.labeled_idxs[nr.randint(len(self.labeled_idxs))]

	def getLabel (self, idx):
		return self.labels[idx]


class kernelAS (genericAS):

	def __init__ (self, params=Parameters()):
		genericAS.__init__ (self, params)

	def initialize(self, Xf):
		"""
		Xf 			--> r x n matrix of feature values for each point.
		"""
		# Save Xf and initialize some of the variables which depend on Xf
		self.Xf = Xf
		self.r, self.n = Xf.shape

		self.unlabeled_idxs = range(self.n)
		self.labels = [-1]*self.n

		# Initialize some parameters and constants which are needed and not yet initialized
		if self.params.sparse:
			self.Ir = ss.eye(self.r)
		else:
			self.Ir = np.eye(self.r)

		self.l = (1-self.params.eta)/self.params.eta
		if self.params.w0 is None:
			self.params.w0 = 1/self.n


		# Set up some of the initial values of some matrices needed to compute D, BDinv, q and f
		B = 1/(1+self.params.w0)*np.ones(self.n)
		D = np.squeeze(Xf.T.dot(Xf.dot(np.ones((self.n,1))))) 
		self.Dinv = 1./D

		if self.params.sparse:
			self.BDinv = ss.diags([np.squeeze(B*self.Dinv)],[0]).tocsr()
		else:
			self.BDinv = np.squeeze(B*self.Dinv)

		self.q = (1-B)*self.params.pi*np.ones(self.n) # Need to update q every iteration

		# Constructing and inverting C
		if self.params.verbose:
			print("Constructing C")
			t1 = time.time()
		if self.params.sparse:
			self.C = (self.Ir - self.Xf.dot(self.BDinv.dot(self.Xf.T)))	
		else:
			self.C = (self.Ir - self.Xf.dot(self.BDinv[:,None]*self.Xf.T))
		if self.params.verbose:
			print("Time for constructing C:", time.time() - t1)

		if self.params.verbose:
			print ("Inverting C")
			t1 = time.time()
		if self.params.sparse:
			self.Cinv = ssl.inv(self.C.tocsc()) # Need to update Cinv every iteration
		else:
			self.Cinv = nlg.inv(self.C)
		if self.params.verbose:
			print("Time for inverse:", time.time() - t1)

	def firstMessage(self, idx):
		# Assuming this is always +ve. Can be changed otherwise
		# Need to check whether this does the right thing.
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

		# Updating various parameters to calculate next C inverse and f
		if self.params.sparse:
			self.BDinv[idx,idx] *= (1+self.params.w0)*self.l/(1+self.l) 
		else:
			self.BDinv[idx] *= (1+self.params.w0)*self.l/(1+self.l) 

		self.q[idx] = lbl*self.l/(1+self.l)
		gamma = -(self.l/(1+self.l)-1/(1+self.params.w0))*self.Dinv[idx]

		Xi = self.Xf[:,[idx]] # ith feature vector
		Cif = self.Cinv.dot(Xi)

		if self.params.sparse:
			self.Cinv = self.Cinv - gamma*(Cif.dot(Cif.T))/(1 + (gamma*Xi.T.dot(Cif))[0,0])
		else:
			self.Cinv = self.Cinv - gamma*(Cif.dot(Cif.T))/(1 + gamma*Xi.T.dot(Cif))
	
		if self.params.sparse:
			self.f = self.q + self.BDinv.dot(((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.q))))))
		else:
			self.f = self.q + self.BDinv*((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.q)))))

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
			print 'Iter: %i, Selected: %i, Hits: %i, Time: %f'%(self.iter, self.labeled_idxs[-1], self.hits[-1], elapsed)
			
	def getStartPoint (self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	# Need to think a bit about the math here
	# def resetLabel (self, idx, lbl):
	# 	raise NotImplementedError()

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