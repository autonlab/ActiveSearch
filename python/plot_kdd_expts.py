
from __future__ import division
import sklearn.feature_extraction.text as sft
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib, matplotlib.pyplot as plt

import time
import csv
import os, os.path as osp
import cPickle as pick
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
alg_names = ['kAS', 'NNAS', 'AGAS']

def get_expts_from_dir (dir_path):
	fnames = os.listdir(dir_path)
	expt_data = []
	for fname in fnames:
		if not osp.isdir(osp.join(dir_path,fname)):
			with open(osp.join(dir_path,fname),'r') as fh: 
				expt_data.append(pick.load(fh))

	hits = {k:[] for k in alg_names}

	for dat in expt_data:
		for k in dat.keys():
			hits[k].append(dat[k])

	hits = {k:np.array(hits[k]) for k in alg_names}
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

def plot_expts (hits, prev=0, max_possible=None, ind_expts=False, title='', save=False):

	num_exp, max_iter = hits[hits.keys()[0]].shape
	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits}
	std_hits = {k:hits[k].std(axis=0).squeeze() for k in hits}
	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}

	itr = range(max_iter)
	chance = (np.array(itr)*prev).tolist()
	if max_possible is None:
		ideal = (np.array(itr)+1).tolist()
	else:
		ideal = range(1, max_possible+1) + [max_possible+1]*(max_iter-max_possible)

	ax = plt.subplot()

	if ind_expts:
		for k in alg_names:
			for run in range(num_exp):
				plt.plot(itr, hits[k][run, :], color=color_map[k], alpha=0.2, linewidth=2)
			plt.plot(itr, mean_hits[k], label=name_map[k], color=color_map[k], linewidth=5)
	else:
		for k in alg_names:
			y1 = mean_hits[k]-std_hits[k]
			y1 = np.where(y1>0, y1, 0)
			y2 = mean_hits[k]+std_hits[k]
			y2 = np.where(y2<ideal, y2, ideal)

			ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=color_map[k], alpha=0.2, interpolate=True)
			ax.plot(itr, y1, color=color_map[k], linewidth=1)
			ax.plot(itr, y2, color=color_map[k], linewidth=1)
			ax.plot(itr, mean_hits[k], label=name_map[k], color=color_map[k], linewidth=5)

	# ideal and random chance
	ax.plot(itr, ideal, 'k', label='Ideal', linewidth=4)
	ax.plot(itr, chance, 'ko', label='Chance', alpha=0.5, linewidth=4)

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


if __name__=='__main__':
	import sys

	## Argument 1: 1/2/3 -- covtype/SUSY/HIGGS
	## Argument 2: 1/2 -- small/large
	## Argument 3: prevalence of +ve class
	## Argument 4: 0/1 projected features?
	## Argument 5: 0/1 save?

	dset = 1
	sl = 2
	prev = 0.47
	proj = False
	save = False

	if len(sys.argv) > 1:
		try:
			dset = int(sys.argv[1])
		except:
			dset = 1
		if dset not in [1,2,3]:
			dset = 1

	if len(sys.argv) > 2:
		try:
			sl = int(sys.argv[2])
		except:
			sl = 1
		if sl not in [1,2]:
			sl = 2

	if len(sys.argv) > 3:
		try:
			prev = float(sys.argv[3])
		except:
			prev = 5.
		if prev < 0 or prev > 5.0:
			prev = 5.0

	if len(sys.argv) > 4:
		try:
			proj = bool(int(sys.argv[4]))
		except:
			proj = False

	if len(sys.argv) > 5:
		try:
			save = bool(int(sys.argv[4]))
		except:
			save = False

	dset_name = {1:'covtype',2:'SUSY',3:'HIGGS'}[dset]
	expt_type = {1:'small',2:'large'}[sl]
	if proj:
		dname = osp.join(results_dir, 'kdd/%s/expts/%s/%.2f/proj/'%(dset_name, expt_type, prev))
	else:
		dname = osp.join(results_dir, 'kdd/%s/expts/%s/%.2f'%(dset_name, expt_type, prev))

	tname = {1:'CoverType',2:'SUSY',3:'HIGGS'}[dset]
	slname = {1:' Small ', 2:' '}[sl]
	if proj:
		title = '%s%swith Projected Features'%(tname, slname)
	else:
		title = '%s%swith Native Features'%(tname, slname)


	hits = get_expts_from_dir(dname)
	plot_expts (hits, prev=prev/100, max_possible=None, ind_expts=False, title=title, save=save)
