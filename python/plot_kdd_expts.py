
from __future__ import division
import sklearn.feature_extraction.text as sft
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib, matplotlib.pyplot as plt

import time
import csv
import os, os.path as osp
import cPickle as pick
import argparse
import IPython

import data_utils as du
import activeSearchInterface as ASI

np.set_printoptions(suppress=True, precision=5, linewidth=100)

data_dir = os.getenv('AS_DATA_DIR')
results_dir = os.getenv('AS_RESULTS_DIR')

name_map = {'kAS': 'Linearized AS', 
			'NNAS': 'Nearest Neighbor AS',
			'AGAS': 'Anchor Graph AS'}
color_map = {'kAS': 'b', 
			'NNAS': 'r',
			'AGAS': 'g'}
linestyle_map = {'kAS': '-', 
				'NNAS': '--',
				'AGAS': ''}
marker_map = {	'kAS': '', 
		    	'NNAS': '',
				'AGAS': 'o'}

alg_names = ['kAS', 'NNAS', 'AGAS']

def get_expts_from_dir (dir_path):
	fnames = os.listdir(dir_path)
	expt_data = []
	for fname in fnames:
		if not osp.isdir(osp.join(dir_path,fname)):
			with open(osp.join(dir_path,fname),'r') as fh: 
				expt_data.append(pick.load(fh))

	hits = {}

	for dat in expt_data:
		for k in dat.keys():
			if k not in hits:
				hits[k] = []
			hits[k].append(dat[k])
	hits = {k:np.array(hits[k]) for k in hits}
	return hits

# def plot_expts (hits, title = '', save=True):

# 	itr = range(hits[hits.keys()[0]].shape[1])
# 	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits}
# 	# mean2_hits = {k:hits[k].mean(axis=1).squeeze() for k in hits}
# 	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}
# 	colors = {k:c for k,c in zip(mean_hits.keys(),['r','b'])}
	
# 	for k in alg_names:
# 		for run in range(hits[k].shape[0]):
# 			plt.plot(itr, hits[k][run, :], color=colors[k], alpha=0.2, linewidth=2)
# 		plt.plot(itr, mean_hits[k], label=name_map[k], color=colors[k], linewidth=5)
	
# 	plt.xlabel('Iterations')
# 	plt.ylabel('Number of Hits')
# 	plt.legend(loc=2)

# 	if title:
# 		plt.title(title, y=1.02, fontsize=40)
# 		# gcf().suptitle(title,fontsize=30)

# 	if save:
# 		fname = osp.join(results_dir, 'imgs', title+'_expt1_plot1.pdf')
# 		fig = plt.figure(1)
# 		fig.set_size_inches(24,14)
# 		plt.savefig(fname, format='pdf', transparent=True, facecolor='w')

# 	plt.show()

def plot_expts (hits, prev=0, stdc=0.25, max_possible=None, ind_expts=False, title='', save=False, ptype='lin'):

	num_exp, max_iter = hits[hits.keys()[0]].shape
	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits}
	std_hits = {k:hits[k].std(axis=0).squeeze() for k in hits}
	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}

	itr = range(max_iter)
	chance = (1 + np.array(itr)*prev).tolist()
	if max_possible is None:
		ideal = (np.array(itr)+1).tolist()
	else:
		ideal = range(1, max_possible+1) + [max_possible+1]*(max_iter-max_possible)

	ax = plt.subplot()

	if ind_expts:
		for k in hits:
			for run in range(num_exp):
				plt.plot(itr, hits[k][run, :], color=color_map[k], alpha=0.2, linewidth=2, marker=marker_map[k], linestyle=linestyle_map[k])
			plt.plot(itr, mean_hits[k], color=color_map[k], label=name_map[k], linewidth=5, marker=marker_map[k], linestyle=linestyle_map[k])
	else:
		for k in hits:
			y1 = mean_hits[k]-stdc*std_hits[k]
			y1 = np.where(y1>1, y1, 1)
			y2 = mean_hits[k]+stdc*std_hits[k]
			y2 = np.where(y2<ideal, y2, ideal)

			ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=color_map[k], alpha=0.2, interpolate=True)
			ax.plot(itr, y1, color=color_map[k], linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
			ax.plot(itr, y2, color=color_map[k], linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
			ax.plot(itr, mean_hits[k], color=color_map[k], label=name_map[k], 
					linewidth=5, marker=marker_map[k], markevery=2, linestyle=linestyle_map[k])

	# ideal and random chance
	ax.plot(itr, ideal, 'k', label='Ideal', linewidth=4)
	ax.plot(itr, chance, 'kx', label='Chance', alpha=0.5, linewidth=4, markevery=2)

	plt.xlabel('Iterations',fontsize=30)
	plt.ylabel('Number of Hits',fontsize=30)
	plt.legend(loc=2, fontsize=25)

	if ptype=='log':
		ax.set_yscale('log')

	if save:
			fname = osp.join(results_dir, 'kdd/imgs', save+'.png')
			fig = plt.figure(1)
			fig.set_size_inches(24,14)
			plt.savefig(fname, format='png', transparent=True, facecolor='w')
	else:
		plt.title(title, y=1.02, fontsize=50)
		# if show:
		plt.show()

if __name__=='__main__':
	matplotlib.rcParams.update({'font.size': 25})

	parser = argparse.ArgumentParser(description='KDD expts.')
	parser.add_argument('--dset', help='dataset', default='covtype', type=str, choices=['covtype', 'SUSY', 'HIGGS'])
	parser.add_argument('--etype', help='expt type', default='main', type=str, choices=['main'])
	parser.add_argument('--ptype', help='plot type', default='lin', type=str, choices=['lin', 'log'])
	parser.add_argument('--prev', help='prevalence of positive class', default=0.05, type=float)
	parser.add_argument('--stdc', help='+/- stdc*stddev in plots', default=0.5, type=float)
	parser.add_argument('--proj', help='use projection', action='store_true')
	parser.add_argument('--save', help='save results', action='store_true')

	args = parser.parse_args()

	dset = args.dset
	etype = args.etype
	ptype = args.ptype
	prev = args.prev
	if prev < 0 or prev > 5.00:
		prev = 5.00
	proj = args.proj
	save = args.save
	stdc = args.stdc
	if stdc < 0 or stdc > 1:
		stdc = 0.5


	if proj:
		dname = osp.join(results_dir, 'kdd/%s/expts/%s/%.2f/proj/'%(dset, etype, prev))
	else:
		dname = osp.join(results_dir, 'kdd/%s/expts/%s/%.2f'%(dset, etype, prev))

	tname = {'covtype':'CoverType','SUSY':'SUSY','HIGGS':'HIGGS'}[dset]
	if proj:
		title = '%s with Projected Features'%(tname)
		if save: save = '%s_proj_prev%.2f'%(dset,prev)
	else:
		title = '%s with Native Features'%(tname)
		if save: save = '%s_prev%.2f'%(dset,prev)

	hits = get_expts_from_dir(dname)
	plot_expts (hits, prev=prev/100, stdc=stdc, max_possible=None, ind_expts=False, title=title, save=save, ptype=ptype)
