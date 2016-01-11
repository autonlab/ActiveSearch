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

name_map = {'AS': 'Active Search - Native Features',
			'ASL': 'Active Search - Learned Features'}

def load_expt_data (fname):

	expt_data = {}
	fh = open(fname,'r')
	for line in fh.readlines():
		exp, itr, hit = [int(i) for i in line.split(',')]
		if exp not in expt_data:
			expt_data[exp] = []
		expt_data[exp].append((itr,hit))
	fh.close()

	return expt_data

def load_all_expts (basename='HIGGS'):

	AS = load_expt_data(osp.join(data_dir, '%s_dummy.csv'%basename))
	ASL = load_expt_data(osp.join(data_dir, '%s.csv'%basename))

	AS_data = []
	ASL_data = []
	for exp in AS:
		AS_data.append(np.cumsum([p[1] for p in AS[exp]]))
		ASL_data.append(np.cumsum([p[1] for p in ASL[exp]]))

	return {'AS': np.array(AS_data), 'ASL': np.array(ASL_data)}

def plot_expts (hits, title='', prev=0, ind_expts=True, save=False, show=False):

	itr = range(hits[hits.keys()[0]].shape[1])
	ideal = (np.array(itr)+1).tolist()
	chance = (np.array(itr)*prev).tolist()
	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits} 
	std_hits = {k:hits[k].std(axis=0).squeeze() for k in hits} 
	# mean2_hits = {k:hits[k].mean(axis=1).squeeze() for k in hits}
	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}

	colors = {k:c for k,c in zip(hits.keys(),['r','b'])}
	
	ax = plt.subplot()

	for k in hits:
		if ind_expts:
			y1 = mean_hits[k]-std_hits[k]
			y1 = np.where(y1>0, y1, 0)
			y2 = mean_hits[k]+std_hits[k]
			y2 = np.where(y2<ideal, y2, ideal)
			ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=colors[k], alpha=0.2, interpolate=True)
			ax.plot(itr, y1, color=colors[k], linewidth=1)
			ax.plot(itr, y2, color=colors[k], linewidth=1)
			# for run in range(hits[k].shape[0]):
			# 	ax.plot(itr, hits[k][run, :], color=colors[k], alpha=0.2, linewidth=2)
		ax.plot(itr, mean_hits[k], label=name_map[k], color=colors[k], linewidth=5)

	# ideal and random chance
	# plt.figure()
	ax.plot(itr, ideal, 'k', label='Ideal', linewidth=5)
	ax.plot(itr, chance, 'ko', label='Chance', linewidth=5)

	plt.xlabel('Iterations',fontsize=50)
	plt.ylabel('Number of Hits',fontsize=50)
	plt.legend(loc=2)

	if title:
		plt.title(title, y=1.02, fontsize=50)
		# gcf().suptitle(title,fontsize=30)

	if save:
		fname = title+'.png'#osp.join(results_dir, 'imgs', title+'_expt1_plot1.pdf')
		fig = plt.figure(1)
		fig.set_size_inches(24,14)
		plt.savefig(fname, format='png', transparent=True, facecolor='w')

	# if show:
	plt.show()

if __name__ == '__main__':
	import sys
	matplotlib.rcParams.update({'font.size': 30})

	higgs = load_all_expts('HIGGS_10exp0.05')
	covertype = load_all_expts('covtype_10exp0.0025')
	susy = load_all_expts('SUSY_10exp0.005')

	plot_expts(higgs, title='Higgs', prev=0.05, save=True)
	plot_expts(covertype, title='CoverType', prev=0.0025, save=True)
	plot_expts(susy, title='SUSY', prev=0.005, save=True)
	# plt.figure()
	# plt.show()