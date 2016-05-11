#! /usr/bin/python
from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def polarToCartesian (r, theta):
	return r*np.array([np.cos(theta), np.sin(theta)])

def cartesianToPolas (x, y):
	return np.array([nlg.norm([x,y]), np.arctan(y,x)])

def createEpsilonGraph (X, eps=1, kernel='rbf', gamma=1):
	## Creates an epsilon graph as follows:
	## create with edges between points with distance < eps
	## edge weights are given by kernel
	
	if kernel not in ['rbf']:
		raise NotImplementedError('This function does not support %s kernel.'%kernel)

	dists = ssd.cdist(X,X)
	eps_neighbors = dists < eps

	if kernel == 'rbf':
		A = eps_neighbors*np.exp(-gamma*dists)

	return A