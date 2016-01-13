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
	for permit in bg_data:
		x = []
		for f in field_features:
			if field_features[f]['type'] == 'categorical':
				z = [0]*len(field_features[f]['mapping'])
				z[field_features[f]['mapping'][permit[f]]] = 1
				x.extend(z)
			elif field_features[f]['type'] == 'numerical':
				# Assuming log vautes
				v = np.max([1.0,float(permit[f])]) # killing the zero values 
				z = field_features[f]['grf'].computeRandomFeatures([np.log(v)])
				x.extend(z)
		X.append(x)
	X = np.array(X).mean(axis=0)

	return X.tolist()

def aggregate_data_into_features (data, field_features, sparse=False):
	"""
	Assuming that the data is of the format:
	{id:[... permits ... ] for id in set_of_ids}
	""" 
	X = []
	LabelMap = {}
	idx = 0
	for bg in data:
		X.append(featurize_blockgroup(data[bg], field_features))
		LabelMap[idx] = bg
		idx += 1

	X = np.array(X).T

	return X, LabelMap

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

	blockgroups, field_features = load_blockgroups_data(field_info, city='seattle')
	X, LabelMap = aggregate_data_into_features(blockgroups, field_features)
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
	
