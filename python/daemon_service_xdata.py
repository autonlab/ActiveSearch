#!/usr/bin/python2.7
from flask import Flask
from flask import Response

import activeSearchInterface as asI
import gaussianRandomFeatures as GRF

import argparse
import sys
import os, os.path as osp
import numpy as np
import pandas as pd
from random import randint

##
# To run this, make sure the permissions are right:
# chmod a+x daemon_service.py 
#
# Then run it:
# ./daemon_service.py
##

### Working with Seattle data currently.
def load_blockgroups_data (field_info, city, dataframes_file, bg_file):
        city_data = pd.read_hdf(dataframes_file, city)
        blockgroups_data = pd.read_hdf(bg_file, 'blockgroups')

        blockgroups = {}
        field_features = {}

        npermits = city_data.shape[0]

	num_nans = 0

        for ipermit in xrange(npermits):
                permit = city_data.iloc[ipermit]
                try:
                        blkg = int(permit.blockgroup)
                except Exception as e:
                        # print ("Warning: Found NaN blockgroup. Ignoring...")
			num_nans += 1
                        continue
                if blkg not in blockgroups:
                        blockgroups[blkg] = []
                blockgroups[blkg].append(permit)

	print ('Ignored %i NaN values in the data.\n'%num_nans)

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


app = Flask(__name__)

parser = argparse.ArgumentParser()
args = parser.parse_args()

activeSearch = None

# when firstMessage is called we reinitialize the kernel algorithm. However calling
# initialize again requires us to invert C so we could be smarter and save that 
# For now the invert time is a couple of seconds so we can do that as future work
restart_save = None
first_run = True

# track the message ID that we're currently presenting the user for evaluation
currentMessage = -1 

data_dir = osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/housing')
bg_file = osp.join(data_dir, 'blockgroups.h5')
dataframes_file = osp.join(data_dir, 'dataframes.h5')

city = 'seattle'
fields = ['category', 'permit_type', 'action_type', 'work_type', 'value']
field_types = ['categorical','categorical','categorical','categorical','numerical']
field_info = {f:ft for f,ft in zip(fields,field_types)}

blockgroups, field_features = load_blockgroups_data (field_info, city, dataframes_file, bg_file)
Xf, BGMap = aggregate_data_into_features (blockgroups, field_features)

BG2IDX = {BGMap[idx]['id']:idx for idx in BGMap}
IDX2BG = {idx:BGMap[idx]['id'] for idx in BGMap}

verbose = True
sparse = False
pi = 0.5
eta = 0.7
prms = asI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
activeSearch = asI.kernelAS(prms)
activeSearch.initialize(Xf)

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

@app.route('/firstmessage/<start_id>')
def firstMessage(start_id):
    #res = randint(0,99)
    #global first_run
    #if (first_run == False and restart_save != None):
        #activeSearch.initialize(restart_save)

    #first_run = False
    activeSearch.firstMessage(BG2IDX[int(start_id)])
    res_idx= activeSearch.getNextMessage()
    res = str(IDX2BG[res_idx]),"\n", format_dict(BGMap[res_idx]["display_data"]), "\n"
    return Response(res,  mimetype='text/plain')
    
@app.route('/messageinteresting')
def interestingMessage():
    activeSearch.interestingMessage()
    res_idx = activeSearch.getNextMessage()
    res = str(IDX2BG[res_idx]),"\n", format_dict(BGMap[res_idx]["display_data"]), "\n"
    return Response(res,  mimetype='text/plain')

@app.route('/messageboring')
def boringMessage():
    activeSearch.boringMessage()
    res_idx = activeSearch.getNextMessage()
    res = str(IDX2BG[res_idx]),"\n", format_dict(BGMap[res_idx]["display_data"]), "\n"
    return Response(res,  mimetype='text/plain')

#@app.route('/getNextMessage')
#def getNextMessage():
#    res = activeSearch.getNextMessage()
#    return Response(str(res),  mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
