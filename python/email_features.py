import numpy as np, scipy as sc
import csv
import datetime

def load_timestamps (tsfile):
	
	ts_data = []

	tsf = open(tsfile, 'r')
	lines = tsf.readlines()

	for line in lines:
		dat,tim = line.split()
		y,mo,d = [int(v) for v in dat.split('-')]
		h,mi,s = [int(v) for v in tim.split(':')]

		ts_data.append(datetime.datetime(y,mo,d,h,mi,s))

	tsf.close()

	return ts_data

def load_tfidf_mat(tffile):

	tsf = open(tsfile, 'r')
	lines = tsf.readlines()

	for line in lines:
		dat,tim = line.split()
		y,mo,d = [int(v) for v in dat.split('-')]
		h,mi,s = [int(v) for v in tim.split(':')]

		ts_data.append(datetime.datetime(y,mo,d,h,mi,s))

	tsf.close()

	return ts_data