import numpy as np

import matplotlib, matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import gcf

import csv
import os, os.path as osp
import cPickle as pick

import IPython

np.set_printoptions(suppress=True, precision=5, linewidth=100)
results_dir = osp.join(os.getenv('HOME'), 'Research/ActiveSearch/results/nips')
# results_dir = '/usr0/home/sibiv/Research/ActiveSearch/results/nips'

colors = [	(1,0,0), (0,1,0), (0,0,1),
		 	(1,1,0), (1,0,1), (0,1,1), 
		 	(1,0.5,0.5), (0.5,1,0.5), (0.5,0.5,1),
		 	(0.2,0.5,0.8) ]

def load_data (hitsfile):

	fh = open(osp.join(results_dir, hitsfile), 'r')
	csvdat = csv.reader(fh)

	init = True
	for dat in csvdat:
		if init:
			keys = [d.split('_')[0] for d in dat[1::2]]
			means = {k:[] for k in keys}
			sds = {k:[] for k in keys}
			init = False
			continue
		for k in xrange(len(keys)):
			means[keys[k]].append(float(dat[2*k+1]))
			sds[keys[k]].append(float(dat[2*k+2]))

	for k in keys:
		means[k] = np.array(means[k])
		sds[k] = np.array(sds[k])

	fh.close()

	return keys, means, sds

def plot_expts (means, sds, keys=None, prev=0, stdc=0.5, max_possible=None, title='', save=False, ptype='lin'):

	import matplotlib as mpl
	fsize = 30

	colors = [	(1,0,0), (0,1,0), (0,0,1),
		 	(1,1,0), (1,0,1), (0,1,1), 
		 	(1,0.5,0.5), (0.5,1,0.5), (0.5,0.5,1),
		 	(0.2,0.5,0.8) ]

	if keys is None:
		keys = means.keys()

	color_map = {keys[k]:colors[k] for k in range(len(keys))}
	marker_map = {k:'' for k in keys}
	linestyle_map = {k:'-' for k in keys}


	max_iter = means[keys[0]].shape[0]
	itr = range(max_iter)
	if prev > 0:
		chance = (1 + np.array(itr)*prev).tolist()

	ax = plt.subplot()

	if 'ideal' in keys:
		ideal = means['ideal']
		ax.plot(itr, ideal, color=color_map['ideal'], label='ideal', linewidth=4)
	else:
		if max_possible is None:
			ideal = (np.array(itr)+1).tolist()
		else:
			ideal = range(1, max_possible+1) + [max_possible+1]*(max_iter-max_possible)
		ax.plot(itr, ideal, 'k', label='Ideal', linewidth=4)
		# ax.plot(itr, ideal, color=color_map['ideal'], label='ideal', linewidth=4)

	for k in keys:
		if k == 'ideal':
			continue
		y1 = means[k]-stdc*sds[k]
		y1 = np.where(y1>1, y1, 1)
		y2 = means[k]+stdc*sds[k]
		if 'ideal' in keys:
			y2 = np.where(y2<ideal, y2, ideal)

		ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=color_map[k], alpha=0.2, interpolate=True)
		ax.plot(itr, y1, color=color_map[k], linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
		ax.plot(itr, y2, color=color_map[k], linewidth=1)#, marker=marker_map[k], linestyle=linestyle_map[k])
		ax.plot(itr, means[k], color=color_map[k], label=k, 
				linewidth=2, marker=marker_map[k], markevery=2, linestyle=linestyle_map[k])

	# ideal and random chance
	# 
	if prev > 0:
		ax.plot(itr, chance, 'kx', label='chance', alpha=0.5, linewidth=4, markevery=2)

	# mpl.rcParams['xtick.labelsize'] = fsize
	# mpl.rcParams['ytick.labelsize'] = fsize
	# import IPython
	# IPython.embed()
	# ax.set_xticklabels(ax.get_xticklabels(), fontsize=fsize)
	# ax.set_yticklabels(ax.get_yticklabels(), fontsize=fsize)
	[tick.label.set_fontsize(int(fsize*0.7)) for tick in ax.xaxis.get_major_ticks()]
	[tick.label.set_fontsize(int(fsize*0.7)) for tick in ax.yaxis.get_major_ticks()]
	plt.xlabel('Iterations',fontsize=fsize)
	plt.ylabel('Number of Hits',fontsize=fsize)
	plt.legend(loc=2, fontsize=fsize)

	if ptype=='log':
		ax.set_yscale('log')

	if save:
		fname = osp.join(results_dir, 'imgs', save+'.png')
		# plt.title(title, y=1.02, fontsize=40)
		fig = plt.figure(1)
		fig.set_size_inches(20,16)
		plt.savefig(fname, format='png', transparent=True, facecolor='w')
	else:

		plt.title(title, y=1.02)#, fontsize=40)
		# if show:
		plt.show()


if __name__ == '__main__':

	prev = 0.05
	# if prev == 0.05:
	# 	fl = 'results_hits.csv'
	# else:
	fl = 'results_hits%.2f.csv'%prev
	title = 'Hits for %.2f'%prev

	k,m,s = load_data(fl)
	print k
	# ks = ['ideal', 'bandit', 'nn', 'eps', 'convex']
	ks = ['ideal', 'nn', 'aew']
	plot_expts (m, s, keys=ks, prev=prev, title=title, save=False)
