import numpy as np

import matplotlib, matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import gcf

import os, os.path as osp
import cPickle as pick

import IPython

np.set_printoptions(suppress=True, precision=5, linewidth=100)

data_dir = os.getenv('AS_DATA_DIR')
results_dir = os.getenv('AS_RESULTS_DIR')
name_map = {'kAS': 'No learning', 
			'aAS_all': 'Learning with all data', 
			'aAS2_recent': 'Learning with recent data'}
alg_names = ['aAS_all', 'aAS2_recent', 'kAS']

def get_expts_from_dir (dir_path):
	fnames = os.listdir(dir_path)

	expt_data = []
	for fname in fnames:
		with open(osp.join(dir_path,fname),'r') as fh: 
			expt_data.append(pick.load(fh))

	hits = {k:[] for k in expt_data[0].keys()}

	for dat in expt_data:
		for k in dat.keys():
			hits[k].append(dat[k])

	hits = {k:np.array(hits[k]) for k in hits.keys()}
	return hits

def plot_expts (hits, title = '', save=True):

	itr = range(hits[hits.keys()[0]].shape[1])
	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits} 
	# mean2_hits = {k:hits[k].mean(axis=1).squeeze() for k in hits}
	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}
	colors = {k:c for k,c in zip(mean_hits.keys(),['r','g','b'])}
	
	for k in alg_names:
		for run in range(hits[k].shape[0]):
			plt.plot(itr, hits[k][run, :], color=colors[k], alpha=0.2, linewidth=2)
		plt.plot(itr, mean_hits[k], label=name_map[k], color=colors[k], linewidth=5)
	
	plt.xlabel('Iterations')
	plt.ylabel('Number of Hits')
	plt.legend(loc=2)
	
	if title:
		plt.title(title, y=1.02, fontsize=40)
		# gcf().suptitle(title,fontsize=30)
	# IPython.embed()
	if save:
		fname = osp.join(results_dir, 'imgs', title+'_expt2_plot1.pdf')
		fig = plt.figure(1)
		fig.set_size_inches(24,14)
		plt.savefig(fname, format='pdf', transparent=True, facecolor='w')
	plt.show()

def plot_subplot_expts(hits, title = '', save=True):

	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits} 
	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}
	colors = {k:c for k,c in zip(mean_hits.keys(),['r','g','b'])}
	itr = range(hits[hits.keys()[0]].shape[1])
	exp = range(hits[hits.keys()[0]].shape[0])
	max_alg = {i:max([max_hits[k][i] for k in max_hits]) for i in exp}
	overall_max = max(max_alg.values())

	if save:
		fig = plt.figure(1)
		fig.set_size_inches(24,14)

	gs = gridspec.GridSpec(5,5)	
	ax = {}
	for i in exp:
		ax[i] = plt.subplot(gs[int(i>=5), i%5])
		ax[i].set_ylim([0,overall_max])
		ax[i].axes.get_xaxis().set_ticks([])
		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)
	ax[10] = plt.subplot(gs[2:,:])
	
	sub_itr = range(200, hits[hits.keys()[0]].shape[1])
	n_sub = len(sub_itr)
	for k in alg_names:
		for run in range(hits[k].shape[0]):
			ax[run].plot(sub_itr, hits[k][run, -n_sub:], color=colors[k], linewidth=3, alpha=0.8)
			ax[run].set_yticks([max_alg[run]])
		ax[10].plot(itr, mean_hits[k], label=name_map[k], color=colors[k], linewidth=5, alpha=0.8)
	
	plt.xlabel('Iterations')
	plt.ylabel('Number of Hits')
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.legend(loc=2)
	if title:
		# plt.title(title)
		gcf().suptitle(title,fontsize=30)
	# IPython.embed()
	if save:
		fname = osp.join(results_dir, 'imgs', title+'_expt2_plot2.pdf')
		plt.savefig(fname, format='pdf', transparent=True, facecolor='w')
	plt.show()


if __name__ == '__main__':
	import sys
	matplotlib.rcParams.update({'font.size': 30})

	expt_dir = osp.join(results_dir,'covertype/run6')
	if len(sys.argv) > 1:
		expt_dir = osp.join(results_dir,sys.argv[1])
		if not osp.isdir(expt_dir):
			expt_dir = osp.join(results_dir,'covertype/run6')

	ptype = 1
	if len(sys.argv) > 2:
		try:
			ptype = int(sys.argv[2])
		except:
			ptype = 1
		if ptype not in [1,2]:
			ptype = 1

	title = ''
	if len(sys.argv) > 3:
		title = sys.argv[3]

	
	hits = get_expts_from_dir(expt_dir)
	if ptype == 1:
		plot_expts(hits, title)
	else:
		plot_subplot_expts(hits, title)
		

	# RUN 6 FOR COVERTYPE SEEMS BEST
	