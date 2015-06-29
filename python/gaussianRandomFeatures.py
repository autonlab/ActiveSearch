from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg

class GaussianRandomFeatures:
	"""
	Class to store Gaussian Random Features.
	"""
	def __init__(self, dim, rn, gammak=1.0, sine=False):
		"""
		Initialize with dim of input space, dim of random feature space
		and bandwidth of the RBF kernel.
		"""
		self.dim = dim
		self.rn = rn
		self.gammak = gammak
		self.sine = sine

		self.generateCoefficients()

	def generateCoefficients (self):
		"""
		Generate coefficients for GFF.
			"""
		self.ws = []
		if not self.sine:
			self.bs = []
		mean = np.zeros(self.dim)
		cov = np.eye(self.dim)*(2*self.gammak)

		if self.sine:
			for _ in range(self.rn):
				self.ws.append(nr.multivariate_normal(mean, cov))
		else:
			for _ in range(self.rn):
				self.ws.append(nr.multivariate_normal(mean, cov))
				self.bs.append(nr.uniform(0.0, 2*np.pi))

	def computeRandomFeatures (self, f):
		"""
		Projects onto fourier feature space.
		"""

		f = np.array(f)
		#f = np.atleast_2d(f)
		ws = np.array(self.ws)
		if self.sine:
			rf_cos = (np.cos(ws.dot(f))*np.sqrt(1/self.rn)).tolist()
			rf_sin = (np.sin(ws.dot(f))*np.sqrt(1/self.rn)).tolist()

			return np.array(rf_cos + rf_sin)
		else:
			bs = np.array(self.bs)
			rf = np.cos(ws.dot(f) + bs[:,None])*np.sqrt(2/self.rn)
			return rf

	def RBFKernel(self, f1, f2, gammak=None):
		"""
		Computes RBF Kernel.
		"""
		if gammak is None: gammak = self.gammak
		
		f1 = np.array(f1)
		f2 = np.array(f2)

		return np.exp(-gammak*(nlg.norm(f1 - f2)**2))

	def LinearRandomKernel(self, f1, f2):
		"""
		Computes Linear Kernel after projecting onto fourier space.
		"""

		rf1 = self.computeRandomFeatures(f1)
		rf2 = self.computeRandomFeatures(f2)

		return np.squeeze(rf1).dot(np.squeeze(rf2))


class RandomFeaturesConverter:

	def __init__(self, dim, rn, gammak, sine=False, feature_generator=None):
		"""
		dim 	--> dimension of input space
		rn  	--> number of random features
		gammak 	--> bandwidth of rbf kernel
		sine 	--> use sin in the random fourier features
		"""

		self.dim = dim
		self.rn = rn
		self.gammak = gammak

		if feature_generator is None:
			self.feature_generator = GaussianRandomFeatures(self.dim, self.rn, self.gammak, sine=sine)
		else: self.feature_generator = feature_generator

	def getFeatureGenerator(self):
		"""
		Get stored feature generator.
		"""
		return self.feature_generator

	def getData (self, fs):
		"""
		Gets the projected features.
		"""
		assert len(fs[0]) == self.dim

		rfs = []
		for f in fs:
			rfs.append(self.feature_generator.computeRandomFeatures(f))

		return rfs