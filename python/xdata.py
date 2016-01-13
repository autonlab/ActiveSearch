from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
# import matplotlib.pyplot as plt
import time
import os, os.path as osp
import csv
import cPickle as pick
# import sqlparse as sql
import json

import adaptiveActiveSearch as AAS
import activeSearchInterface as ASI
import similarityLearning as SL
import data_utils as du

import IPython

data_dir = osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/housing/formatted-data')

def extract_fields (dat, fields):
	extracted_data = []
	for d in dat:
		ed = {f:d[f] for f in fields}
		extracted_data.append(ed)
	return extracted_data

if __name__ == '__main__':
	fields = [u'category','value', 'permit_type', 'work_type']
	with open(osp.join(data_dir, 'mags-97de.json' ),'r') as fh:
		seattle_data = json.load(fh)
	ed = extract_fields (seattle_data[seattle_data.keys()[0]], fields)
	for e in ed:
		e['value'] = float(e['value'])	
	labels = {f:set([e[f] for e in ed]) for f in fields}

	IPython.embed()
	
