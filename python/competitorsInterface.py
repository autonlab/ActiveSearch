from __future__ import division, print_function
import time
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg

import activeSearchInterface as ASI

np.set_printoptions(suppress=True, precision=5, linewidth=100)


def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))

class NNParameters (ASI.Parameters):

	def __init__ (self, normalize=True, sparse=True, verbose=True):
		"""
		Parameters specifically for nearest neighbours.
		"""
		ASI.Parameters.__init__ (self, sparse=sparse, verbose=verbose)
		self.normalize = normalize

class averageNN (ASI.genericAS):

	def __init__ (self, params=NNParameters()):
		ASI.genericAS.__init__ (self, params)

	def initialize(self, Xf, init_labels = {}):
		"""
		Xf 			--> r x n matrix of feature values for each point.
						where r is # features, n is # points.
		init_labels	--> dictionary from emailID to label of initial labels.
		"""
		# Save Xf and initialize some of the variables which depend on Xf
		self.Xf = Xf
		self.r, self.n = Xf.shape

		self.labeled_idxs = init_labels.keys()
		self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

		self.labels = np.array([-1]*self.n)
		self.labels[self.labeled_idxs] = init_labels.values()

		self.NN_similarity = None
		if self.params.normalize is True:
			self.abs_NN_similarity = None

		# Setting iter/start_point
		# If batch initialization is done, then start_point is everything given
		if len(self.labeled_idxs) > 0:
			if len(self.labeled_idxs) == 0:
				self.start_point = self.labeled_idxs[0]
			else:
				self.start_point = [eid for eid in self.labeled_idxs]

			## Computing KNN similarity
			NN_similarity = self.Xf[:, self.unlabeled_idxs].T.dot(self.Xf[:, self.labeled_idxs])
			if self.params.normalize:
				self.NN_avg_similarity = self.NN_similarity.dot(self.labels[self.labeled_idxs]).squeeze()
				self.NN_abs_similarity = np.array(np.abs(NN_similarity).sum(1)).squeeze()
				self.f = self.NN_avg_similarity*(1/self.NN_abs_similarity)
			else:
				self.f = self.NN_similarity.dot(self.labels[self.labeled_idxs]).squeeze()

			# Finding the next message to show -- get the current max element
			self.uidx = np.argmax(self.f)
			self.next_message = self.unlabeled_idxs[self.uidx]
			# Now that a new message has been selected, mark it as unseen
			self.seen_next = False 

			self.iter = 0
			self.hits = [sum(init_labels.values())]

		if self.params.verbose:
			print("Done with the initialization.")

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

	def setLabel (self, idx, lbl, display_iter = None):
		# Set label for given message id
		# Always assuming idx has not been labeled yet.

		if self.params.verbose:
			t1 = time.time()

		# just in case, force lbl to be 0 or 1
		lbl = 0 if lbl <= 0 else 1
	
		# First, some book-keeping
		# If setLabel is called without "firstMessage," then set start_point
		if self.start_point is None:
			self.start_point = idx
		if self.next_message != idx:
			self.uidx = np.nonzero(np.array(self.unlabeled_idxs)==idx)[0][0]

		self.iter += 1
		self.labels[idx] = lbl
		self.unlabeled_idxs.remove(idx)

		# Updating various parameters to calculate f
		Xi = self.Xf[:,[idx]] # ith feature vector
		Sif = self.Xif[:, self.unlabeled_idxs].T.dot(Xi)

		if self.params.normalize:
			self.NN_avg_similarity = np.delete(self.NN_avg_similarity, self.uidx) + lbl*Sif
			self.NN_abs_similarity = np.delete(self.NN_abs_similarity, self.uidx) + np.abs(Sif)
			self.f = self.NN_avg_similarity*(1/self.abs_NN_similarity)
		else:
			self.f = np.delete(self.f, self.uidx) + lbl*Sif

		# Some more book-keeping
		self.labeled_idxs.append(idx)
		if self.iter == 0:
			self.hits.append(lbl)
		else:
			self.hits.append(self.hits[-1] + lbl)

		# Finding the next message to show -- get the current max element
		self.uidx = np.argmax(self.f)
		self.next_message = self.unlabeled_idxs[self.uidx]
		# Now that a new message has been selected, mark it as unseen
		self.seen_next = False 

		if self.params.verbose:
			elapsed = time.time() - t1
			display_iter = display_iter if display_iter else self.iter
			print( 'Iter: %i, Selected: %i, Hits: %i, Time: %f'%(display_iter, self.labeled_idxs[-1], self.hits[-1], elapsed))
			
	def getStartPoint (self):
		if self.start_point is None:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.start_point

	def resetLabel (self, idx, lbl):
		# Reset label for given message id

		# If reset label is called on something not yet set, it should do the same as setLabel
		ret = self.labels[idx]
		if self.labels[idx] == -1:
			self.setLabel(idx, lbl)
			return ret 
		elif self.labels[idx] == lbl:
			print("Already the same value!")
			return ret

		if self.params.verbose:
			t1 = time.time()

		# just in case, force lbl to be 0 or 1
		lbl = 0 if lbl <= 0 else 1
	
		# Updating various parameters to calculate and f
		Xi = self.Xf[:,[idx]] # ith feature vector
		Sif = self.Xif[:, self.unlabeled_idxs].T.dot(Xi)

		if self.params.normalize:
			self.NN_avg_similarity += 2*lbl*Sif
			self.f = self.NN_avg_similarity*(1/self.abs_NN_similarity)
		else:
			self.f += 2*lbl*Sif

		# First, some book-keeping
		# Message is already labeled, se we don't change self.unlabeled_idx
		self.iter += 1 # We're increasing iter because it will be consistent with "hits"
		self.labels[idx] = lbl

		# Some more book-keeping
		# Not sure how we're going to change "history" in some sense:
		# -- everything since this was first incorrectly labeled.
		self.hits.append(self.hits[-1] + (-1 if lbl == 0 else 1))

		# Finding the next message to show -- get the current max element
		uidx = np.argmax(self.f)
		self.next_message = self.unlabeled_idxs[self.uidx]
		# Now that a new message has been selected, mark it as unseen
		self.seen_next = False 

		if self.params.verbose:
			elapsed = time.time() - t1
			print('Iter: %i, Selected: %i, Hits: %i, Time: %f'%(self.iter, self.labeled_idxs[-1], self.hits[-1], elapsed))

		return ret

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
			self.setLabel(idx,lbl)

	def pickRandomLabeledMessage (self):
		if iter < 0:
			raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
		return self.labeled_idxs[nr.randint(len(self.labeled_idxs))]

	def getLabel (self, idx):
		# 0 is boring, 1 is interesting and -1 is unlabeled
		return self.labels[idx]