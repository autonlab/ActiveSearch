from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss

import time
import os, os.path as osp
import json, h5py, pandas as pd

import activeSearchInterface as ASI
import gaussianRandomFeatures as GRF

import IPython

data_dir = osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/housing')

### Working with Seattle data currently.
def load_blockgroups_data (field_info, city='seattle'):
	city_data = pd.read_hdf(osp.join(data_dir,'dataframes.h5'), city)
	blockgroups_data = pd.read_hdf(osp.join(data_dir,'blockgroups.h5'), 'blockgroups')

	blockgroups = {}
	field_features = {}

	npermits = city_data.shape[0]

	for ipermit in xrange(npermits):
		permit = city_data.iloc[ipermit]
		try:
			blkg = int(permit.blockgroup)
		except Exception as e:
			print ("Warning: Found NaN blockgroup. Ignoring...")
			break
		if blkg not in blockgroups:
			blockgroups[blkg] = []
		blockgroups[blkg].append(permit)

	for f in field_info:
		field_features[f] = {'type':field_info[f]}
		if field_info[f] == 'categorical':
			field_features[f]['mapping'] = {}
			for i,fnam in enumerate(np.unique(city_data[f].values)):
				field_features[f]['mapping'][fnam] = i
		elif field_info[f] == 'numerical':
			field_features[f]['grf'] = GRF.GaussianRandomFeatures(dim=1, gammak=0.25, rn=50, sine=True)

	return blockgroups, field_features

def featurize_blockgroup (bg_data, field_features):
	"""
	Featurize the permits based on the relevant fields.
	Some fields will just end up as binary bit arrays (categorical fields).
	Some others will appear as gaussian random features.

	Then averages over all these for the blockgroup.
	"""
	X = []
	field_data = {f:[] for f in field_features}
	for permit in bg_data:
		x = []
		for f in field_features:
			if field_features[f]['type'] == 'categorical':
				z = [0]*len(field_features[f]['mapping'])
				z[field_features[f]['mapping'][permit[f]]] = 1
				x.extend(z)
				field_data[f].append(z)

			elif field_features[f]['type'] == 'numerical':
				# Assuming log vautes
				v = np.max([1.0,float(permit[f])]) # killing the zero values 
				z = field_features[f]['grf'].computeRandomFeatures([np.log(v)])
				x.extend(z)
				field_data[f].append(v)
		X.append(x)
	
	X = np.array(X).mean(axis=0)
	field_avg_data = {}
	for f in field_data:
		if field_features[f]['type'] == 'categorical':
			f_dict = {}
			Xf = np.array(field_data[f]).mean(axis=0).tolist()
			for c,idx in field_features[f]['mapping'].items():
				f_dict[c] = Xf[idx]
			field_avg_data[f] = f_dict
		elif field_features[f]['type'] == 'numerical':
			f_dict = {}
			Xf = field_data[f]
			f_dict['mean'] = np.mean(Xf)
			f_dict['median'] = np.median(Xf)
			f_dict['std'] = np.std(Xf)
			field_avg_data[f] = f_dict

	return X.tolist(), field_avg_data

def aggregate_data_into_features (data, field_features, sparse=False):
	"""
	Assuming that the data is of the format:
	{id:[... permits ... ] for id in set_of_ids}
	""" 
	X = []
	BGMap = {}
	idx = 0
	for bg in data:
		Xf, fad = featurize_blockgroup(data[bg], field_features)
		BGMap[idx] = {'id': bg, 'display_data':fad}
		X.append(Xf)
		idx += 1

	X = np.array(X).T

	return X, BGMap

def format_dict(d, indent = 0):
    res = ""
    for key in d:
        res += ("   " * indent) + str(key) + ":\n"
        if not isinstance (d[key], dict):
            res += ("   " * (indent + 1)) + str(d[key]) + "\n"
        else:
            indent += 1
            res += format_dict(d[key], indent)
            indent -= 1
    return res+"\n"

def display_blockgroup(bg_info):
	print (format_dict(bg_info))


def extract_fields (dat, fields):
	extracted_data = []
	for d in dat:
		ed = {f:d[f] for f in fields}
		extracted_data.append(ed)
	return extracted_data


def test_seattle ():
	fields = ['category', 'permit_type', 'action_type', 'work_type', 'value']
	field_types = ['categorical','categorical','categorical','categorical','numerical']
	field_info = {fields[i]:field_types[i] for i in xrange(len(fields))}

	t1 = time.time()
	blockgroups, field_features = load_blockgroups_data(field_info, city='seattle')
	print ('Time taken to load blockgroup data: %.2fs'%(time.time()-t1))
	t1 = time.time()
	X, BGMap = aggregate_data_into_features(blockgroups, field_features)
	print ('Time taken to generate features: %.2fs'%(time.time() - t1))
	IPython.embed()

	# Active search with the features
        verbose = True
        sparse = False
        pi = 0.5
        eta = 0.7
        K = 50

	d,n = X.shape

        # Run Active Search
        prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
        kAS = ASI.kernelAS (prms)

        init_pt = input('Choose initial blockgroup index (1 - %i): '%n)-1
	display_blockgroup(BGMap[init_pt]['display_data'])
        kAS.initialize(X, init_labels={init_pt:1})

        hits = [1]

        for i in xrange(K):

                idx = kAS.getNextMessage()

		display_blockgroup(BGMap[idx]['display_data'])
		yn = raw_input('\nIs this blockgroup of interest to you (y/n)? ')
		if yn.lower() in  ['q','quit']:
			print('\n Quitting Active Search loop.')
			break
		y = 1 if yn.lower() in ['y','yes'] else 0

                kAS.setLabelCurrent(y)

                hits.append(hits[-1]+y)
	
	IPython.embed()


if __name__ == '__main__':

	test_seattle()
	# # fields = [u'category','value', 'permit_type', 'work_type']
	# with open(osp.join(data_dir, 'formatted-data/mags-97de.json' ),'r') as fh:
	# 	seattle_data = json.load(fh)
	# ed = extract_fields (seattle_data[seattle_data.keys()[0]], fields)
	# for e in ed:
	# 	e['value'] = float(e['value'])	
	# labels = {f:set([e[f] for e in ed]) for f in fields}

	# IPython.embed()
	
