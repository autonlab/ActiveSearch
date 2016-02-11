
from __future__ import division
import sklearn.feature_extraction.text as sft
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib, matplotlib.pyplot as plt

import time
import csv
import os, os.path as osp
import IPython

import data_utils as du
import activeSearchInterface as ASI

np.set_printoptions(suppress=True, precision=5, linewidth=100)

data_dir = os.getenv('AS_DATA_DIR')
results_dir = os.getenv('AS_RESULTS_DIR')

name_map = {'kAS': 'Kernel Active Search', 
			'NNAS': 'Learned Features'}
alg_names = ['aAS', 'kAS']
knn_names = ['knn_learned', 'knn_native']

def get_expts_from_dir (dir_path):
	fnames = os.listdir(dir_path)

	expt_data = []
	for fname in fnames:
		with open(osp.join(dir_path,fname),'r') as fh: 
			expt_data.append(pick.load(fh))

	hits = {k:[] for k in alg_names}
	knns = {k:[] for k in knn_names}

	for dat in expt_data:
		for k in dat.keys():
			if k in alg_names:
				hits[k].append(dat[k])
			else:
				knns[k].append(dat[k])

	hits = {k:np.array(hits[k]) for k in alg_names}
	knns = {k:np.array(knns[k]) for k in knn_names}
	return hits, knns

def plot_expts (hits, prev=0, max_possible=None, ind_expts=True, title='', save=False):

	num_exp, max_iter = hits.shape

	itr = range(max_iter)
	chance = (np.array(itr)*prev).tolist()
	if max_possible is None:
			ideal = (np.array(itr)+1).tolist()
	else:
			ideal = range(1, max_possible+1) + [max_possible+1]*(max_iter-max_possible)

	mean_hits = hits.mean(axis=0).squeeze()
	std_hits = hits.std(axis=0).squeeze()
	max_hits = hits.max(axis=1).squeeze()
	# mean2_hits = hits.mean(axis=1).squeeze()
	ax = plt.subplot()

	color = 'b'
	if ind_expts:
			for run in range(num_exp):
					ax.plot(itr, hits[run, :], color=color, alpha=0.2, linewidth=2)
	else:
			y1 = mean_hits-std_hits
			y1 = np.where(y1>0, y1, 0)
			y2 = mean_hits+std_hits
			y2 = np.where(y2<ideal, y2, ideal)
			ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=color, alpha=0.2, interpolate=True)
			ax.plot(itr, y1, color=color, linewidth=1)
			ax.plot(itr, y2, color=colors, linewidth=1)
	ax.plot(itr, mean_hits, label='ActiveSearch', color=color, linewidth=5)

	# ideal and random chance
	ax.plot(itr, ideal, 'k', label='Ideal', linewidth=4)
	ax.plot(itr, chance, 'ro', label='Chance', alpha=0.5, linewidth=4)

	plt.xlabel('Iterations',fontsize=50)
	plt.ylabel('Number of Hits',fontsize=50)
	plt.legend(loc=1)

	if title:
			plt.title(title, y=1.02, fontsize=50)

	if save:
			fname = title+'.png'
			fig = plt.figure(1)
			fig.set_size_inches(24,14)
			plt.savefig(fname, format='png', transparent=True, facecolor='w')

	# if show:
	plt.show()
