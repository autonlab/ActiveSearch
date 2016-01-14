#!/usr/bin/python2.7
from flask import Flask
from flask import Response
import activeSearchInterface as asI
import argparse
import sys
from random import randint

##
# To run this, make sure the permissions are right:
# chmod a+x daemon_service.py 
#
# Then run it:
# ./daemon_service.py
##

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


verbose = True
sparse = False
pi = 0.5
eta = 0.7
prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
activeSearch = asI.kernelAS(prms)

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

@app.route('/firstmessage/<start_id>')
def firstMessage(start_id):
    #res = randint(0,99)
    #global first_run
    #if (first_run == False and restart_save != None):
    activeSearch.initialize(restart_save)

    #first_run = False
    #activeSearch.firstMessage(int(message))
    #res = getNextMessage()
    return Response(str(res),  mimetype='text/plain')
    
@app.route('/messageinteresting')
def interestingMessage():
    res = randint(0,99)
    #activeSearch.interestingMessage()
    #res = activeSearch.getNextMessage()
    return Response(str(res),  mimetype='text/plain')

@app.route('/messageboring')
def boringMessage():
    res = randint(0,99)
    #activeSearch.boringMessage()
    #res = activeSearch.getNextMessage()
    return Response(str(res),  mimetype='text/plain')

#@app.route('/getNextMessage')
#def getNextMessage():
#    res = activeSearch.getNextMessage()
#    return Response(str(res),  mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=False)
