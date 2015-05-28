from __future__ import print_function, division
import sys
import numpy as np
import scipy as sc, scipy.sparse as ss
import csv
import datetime

import gaussianRandomFeatures as grf

def load_timestamps (tsfile):
	
	ts_data = []

	tsf = open(tsfile, 'r')
	lines = tsf.readlines()
	nl = len(lines)

	print ('Loading timestamps...')
	idx = 0
	for line in lines:
		dat,tim = line.split()
		y,mo,d = [int(v) for v in dat.split('-')]
		h,mi,s = [int(v) for v in tim.split(':')]

		# print(y,mo,d)
		# print(h,mi,s)
		try:
			ts_data.append(datetime.datetime(y,mo,d,h,mi,s))
		except:
			ts_data.append([]) # weird 0 time artifact

		idx += 1
		print ('Progress: %f'%(idx/nl*100), end='\r')
		sys.stdout.flush()
	
	print ('Progress: %f'%100)

	tsf.close()
	return ts_data

def create_coo_matrix (row, col, data, shape):

	row = np.squeeze(np.array(row))
	col = np.squeeze(np.array(col))
	data = np.squeeze(np.array(data))

	return ss.coo_matrix((data, (row,col)), shape=shape, dtype='f')

def load_tfidf_data(tffile, as_coo=True):

	tf_rows = []
	tf_cols = []
	tf_data = []

	tdf = open(tffile, 'r')
	lines = tdf.readlines()
	nl = len(lines)

	numw = 0

	print ('Loading tfidf data...')
	idx = 0


	for line in lines:
		eid, wid, v = line.split()
		eid = int(eid)
		wid = int(wid)
		v = float(v)

		tf_rows.append(wid)
		tf_cols.append(eid)
		tf_data.append(v)
		numw = numw if numw > wid else wid + 1

		idx += 1
		print ('Progress: %f'%(idx/nl*100), end='\r')
		sys.stdout.flush()

	print ('Progress: %f'%100)
	nume = eid + 1

	# print ('Progress:   Done.')
	tdf.close()

	if as_coo:
		return create_coo_matrix(tf_rows, tf_cols, tf_data, (numw,nume))
	return tf_rows, tf_cols, tf_data, nume, numw

def load_sender_data (senderfile, as_coo=True):

	sender_rows = []
	sender_cols = []
	sender_data = []

	sf = open(tsfile, 'r')
	lines = sf.readlines()
	nl = len(lines)

	nsender = len(lines[0].split())

	print ('Loading sender data...')
	idx = 0

	for line in lines:
		svals = [int(v) for v in line.split()]
		rows = np.nonzero(svals)
		cols = [idx]*(len(rows))

		sender_rows.extend(rows)
		sender_cols.append(eid)

		idx += 1
		print ('Progress: %f'%(idx/nl*100), end='\r') 
		sys.stdout.flush()

	nume = idx
	print ('Progress: %f'%100)
	sf.close()

	sender_data = np.ones(len(sender_rows))

	if as_coo:
		return create_coo_matrix(sender_rows, sender_cols, sender_data, (nsender,nume))
	return sender_rows, sender_cols, sender_data, nsender, nume



ts_magic_number = 13168189440000.0

def generate_features (tf_F, ts_F=None, ts_rf=100, s_F=None):
	"""
	Create feature matrix from tfidf data and other data.
	"""

	ts_D = None
	if ts_F is not None:
		t0 = datetime.datetime(1970,1,1)
		for ts in ts_F:
			ts_D.append((ts-t0).total_seconds())

		
	print ('Progress:   Done.')
	return TFmat