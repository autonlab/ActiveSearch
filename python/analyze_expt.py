
from __future__ import division
import sklearn.feature_extraction.text as sft
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss

import time
import csv
import os, os.path as osp
import cPickle as pick
import argparse
import IPython

import data_utils as du
import activeSearchInterface as ASI

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def get_expts_from_dir (dir_path):
	fnames = os.listdir(dir_path)
	fnames_proj = os.listdir(osp.join(dir_path,'proj'))
	expt_data = []
	expt_data_proj = []

	for fname in fnames:
		if not osp.isdir(osp.join(dir_path,fname)):
			with open(osp.join(dir_path,fname),'r') as fh: 
				expt_data.append(pick.load(fh))
	for fname in fnames_proj:
		if not osp.isdir(osp.join(dir_path,'proj',fname)):
			with open(osp.join(dir_path,'proj',fname),'r') as fh: 
				expt_data_proj.append(pick.load(fh))

	hits = {}

	for dat in expt_data:
		for k in dat.keys():
			if k not in hits:
				hits[k] = []
			hits[k].append(dat[k])

	for dat in expt_data_proj:
		for k in dat.keys():
			kL = k+'+L'
			if kL not in hits:
				hits[kL] = []
			hits[kL].append(dat[k])

	hits = {k:np.array(hits[k]) for k in hits}
	
	return hits

def analyze_expts (hits, itrs, prev):

	num_exp, max_iter = hits[hits.keys()[0]].shape
	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits}
	std_hits = {k:hits[k].std(axis=0).squeeze() for k in hits}

	order = ['kAS','NNAS','AGAS','kAS+L','NNAS+L','AGAS+L']

	i = 0
	for itr in itrs:
		if i == 0:
			s = '\n%.2f & %i'%(prev, itr)
			i += 1
		else:
			s = '\n & %i'%(itr)
		for o in order:
			s += ' & %.1f $\pm$ %.1f'%(mean_hits[o][itr-1], std_hits[o][itr-1])
		s += '\\\\'
		print (s)

def analyze_expts_all_prevs (dset, itrs, prevs):


	order = ['kAS','NNAS','AGAS','kAS+L','NNAS+L','AGAS+L']

	s = '\hline'
	for prev in prevs:
		dname = osp.join(results_dir, 'kdd/%s/expts/main/%.2f'%(dset, prev))
		hits = get_expts_from_dir(dname)

		num_exp, max_iter = hits[hits.keys()[0]].shape
		mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits}
		std_hits = {k:hits[k].std(axis=0).squeeze() for k in hits}
		print 'here'
		i = 0
		for itr in itrs:
			if i == 0:
				s += '\n%.2f & %i'%(prev, itr)
				i += 1
			else:
				s += '\n & %i'%(itr)
			for o in order:
				s += ' & %.1f $\pm$ %.1f'%(mean_hits[o][itr-1], std_hits[o][itr-1])
			s += '\\\\'
		s += '\n\hline'

	print (s)




if __name__=='__main__':

	parser = argparse.ArgumentParser(description='KDD expts.')
	parser.add_argument('--dset', help='dataset', default='covtype', type=str, choices=['covtype', 'SUSY', 'HIGGS'])
	parser.add_argument('--prev', help='prevalence of positive class', default=0.05, type=float)

	args = parser.parse_args()

	dset = args.dset
	prev = args.prev
	if prev < 0 or prev > 5.00:
		prev = 5.00

	# if proj:
	results_dir = du.results_dir
	dname = osp.join(results_dir, 'kdd/%s/expts/main/%.2f'%(dset, prev))
	# else:
	# 	dname = osp.join(results_dir, 'kdd/%s/expts/%s/%.2f'%(dset, etype, prev))

	hits = get_expts_from_dir(dname)
	# analyze_expts(hits, itrs=[100,200], prev=prev)
	analyze_expts_all_prevs(dset='HIGGS', itrs=[100,200], prevs=[5.,2.5,1.])
