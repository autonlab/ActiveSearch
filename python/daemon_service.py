#!/usr/bin/python2.7
from flask import Flask
from flask import Response
import numpy as np
import mysql_connect as mysql_conn
import activeSearchInterface as asI
import argparse
import sys
import multiprocessing

##
# To run this, make sure the permissions are right:
# chmod a+x daemon_service.py 
#
# Then run it:
# ./daemon_service.py
##

app = Flask(__name__)

cpu_count = multiprocessing.cpu_count()
cpu_count /= 2 #we don't want hyperthreading cores, only physical cores. AMD also kind of cheats by having one FPU per two cores so this seems reasonable as a default

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', default='kernel', help='shari, naiveshari, kernel, default to kernel')
parser.add_argument('-w', '--wordlimit', default=6000, type=int, help='in kernel mode, max number of words to retain. Higher for better accuracy, fewer for better speed. 0=retain all')
parser.add_argument('-d', '--database', default='jebbush', help='database name')
parser.add_argument('-u', '--database_user', default='root', help='database user')
parser.add_argument('-p', '--database_pass', default='', help='database pass')
parser.add_argument('-n', '--database_hostname', default='', help='database hostname')
parser.add_argument('-j', '--JSON_path', default='', help='path to file or directory with JSON flat file twitter data')
parser.add_argument('-z', '--num_cpus', default=cpu_count, type=int, help='number of cpus for tfidf - physical cores only. Default is ' + str(cpu_count))
parser.add_argument('-s', '--skip_stemmer', default=False, action='store_true', help='skip a slow part of tfidf. Drops result quality but improves speed. Save time when testing code')

args = parser.parse_args()

#message_count = dataConn.connect("/home/tw/tweets/pwfiregrl97.tweets")
if (args.JSON_path is not ''):
    dataConn = mysql_conn.flatfileDataConnect()
    message_count = dataConn.connect(args.JSON_path)
else:
    dataConn = mysql_conn.mysqlDataConnect()
    message_count = dataConn.connect(args.database, args.database_hostname, args.database_user, args.database_pass)

activeSearch = None

# when firstMessage is called we reinitialize the kernel algorithm. However calling
# initialize again requires us to invert C so we could be smarter and save that 
# For now the invert time is a couple of seconds so we can do that as future work
restart_save = None
first_run = True
if (args.method == "kernel"):
    print "Using kernelAS"
    activeSearch = asI.kernelAS()
    wMat = dataConn.getFinalFeatureMatrix(args.wordlimit,args.skip_stemmer, args.num_cpus, message_count, 0,0)
    restart_save = wMat.copy()
    activeSearch.initialize(wMat)
elif (args.method == "shari"):
    print "Using shariAS"
    activeSearch = asI.shariAS()   
    A = dataConn.getAffinityMatrix(args.wordlimit,args.skip_stemmer,args.num_cpus, message_count, 0,0)
    # Feeding in the dense version to shari's code because the sparse version is not implemented 
    activeSearch.initialize(np.array(A.todense())) 
elif (args.method == "naiveshari"):
    print "Using naieveShariAS"
    activeSearch = asI.naiveShariAS()   
    A = dataConn.getAffinityMatrix(args.wordlimit,args.skip_stemmer,args.num_cpus, message_count, 0,0)
    # Feeding in the dense version to shari's code because the sparse version is not implemented 
    activeSearch.initialize(np.array(A.todense())) 
else:
    print "Invalid method argument. See help (run with -h)"
    sys.exit()

# track the message ID that we're currently presenting the user for evaluation
currentMessage = -1 

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

## functions that we had implemented in the Java version for TK's version of active search
@app.route('/firstmessage/<message>')
def firstMessage(message):
    global first_run
    if (first_run == False and restart_save != None):
        activeSearch.initialize(restart_save)

    first_run = False
    activeSearch.firstMessage(int(message))
    return Response("ok",  mimetype='text/plain')

@app.route('/messageinteresting')
def interestingMessage():
    activeSearch.interestingMessage()
    res = activeSearch.getNextMessage()
    return Response(str(res),  mimetype='text/plain')

@app.route('/messageboring')
def boringMessage():
    activeSearch.boringMessage()
    res = activeSearch.getNextMessage()
    return Response(str(res),  mimetype='text/plain')

@app.route('/setalpha/<alpha>')
def setalpha(alpha):
    activeSearch.setalpha(float(alpha))
    return Response("ok",  mimetype='text/plain')

@app.route('/getStartPoint')
def getStartPoint():
    res = activeSearch.getStartPoint()
    return Response(str(res),  mimetype='text/plain')

@app.route('/resetLabel/<index>/<value>')
def resetLabel(index, value):
    ret = activeSearch.resetLabel(int(index), int(value))
    return Response(str(ret),  mimetype='text/plain')

@app.route('/setLabelCurrent/<value>')
def setLabelCurrent(value):
    activeSearch.setLabelCurrent(int(value))
    #res = activeSearch.getNextMessage()
    return Response("ok",  mimetype='text/plain')

# input is [index, value [,index, value etc]]
@app.route('/setLabelBulk/<csv>')
def setLabeLBulk(csv):
    idxs = []
    lbls = []
    offset = 0
    for row in csv:
        if (offset == 0):
            offset = 1
            idxs.append(int(row))
        else:
            offset = 0
            lbls.append(int(row))

    if (offset == 1):
        print "Error: odd number of inputs in CSV. Expected index, label pairs\n"
        raise Exception()

    activeSearch.setLabelBulk(idxs, lbls)
    return Response("hello",  mimetype='text/plain')

@app.route('/getNextMessage')
def getNextMessage():
    res = activeSearch.getNextMessage()
    return Response(str(res),  mimetype='text/plain')

@app.route('/pickRandomLabeledMessage')
def pickRandomLabeledMessage():
    res = activeSearch.pickRandomLabeledMessage()
    return Response(str(res),  mimetype='text/plain')

@app.route('/getLabel/<message>')
def getLabel(message):
    res = activeSearch.getLabel(int(message))
    return Response(str(res),  mimetype='text/plain')

#####
# For documentation on the following functions, see their analogs in mysql_connect.py
#####

@app.route('/getUserNameFromID/<id>')
def getUserNameFromID(id):
    return Response(dataConn.getUserNameFromID(int(id)), mimetype='text/plain')

@app.route('/getMessagesFromUserToUser/<from_id>/<to_id>')
def getMessagesFromUserToUser(from_id, to_id):
    return Response(dataConn.getMessagesFromUserToUser(int(from_id), int(to_id)), mimetype='text/plain')

@app.route('/getMessageSubjectFromMessageID/<id>')
def getMessageSubjectFromMessageID(id):
    return Response(dataConn.getMessageSubjectFromMessageID(int(id)), mimetype='text/plain')

@app.route('/getMessageBodyFromMessageID/<id>')
def getMessageBodyFromMessageID(id):
    ret = dataConn.getMessageBodyFromMessageID(int(id))
    mystr = str(ret[0]) + "\n\n" + ret[1] + "\n\n" + ret[2]
    return Response(mystr, mimetype='text/plain')

@app.route('/getTotalMessageCount')
def getTotalMessageCount():
    return Response(str(dataConn.getTotalMessageCount()), mimetype='text/plain')

@app.route('/getMessageTimesAndSenders/<id>')
def getMessageTimesAndSenders(id):
    return Response(dataConn.getMessageTimesAndSenders(), mimetype='text/plain')

@app.route('/getUsersByMessage/<id>')
def getUsersByMessage(id):
    ret_arr = dataConn.getUsersByMessage(int(id))
    mystr = ""
    for row in ret_arr:
        mystr += str(row) + "\n"
    return Response(mystr, mimetype='text/plain')

@app.route('/getSenderByMessage/<id>')
def getSenderByMessage(id):
    return Response(str(dataConn.getSenderByMessage(int(id))), mimetype='text/plain')

@app.route('/getTimeByMessage/<id>')
def getTimeByMessage(id):
    return Response(dataConn.getTimeByMessage(int(id)), mimetype='text/plain')

@app.route('/getSubjectByMessage/<id>')
def getSubjectByMessage(id):
    return Response(dataConn.getSubjectByMessage(int(id)), mimetype='text/plain')

@app.route('/getMessagesByKeyword/<word>')
def getMessagesByKeyword(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = dataConn.getMessagesByKeyword(word)
    mystr = ""
    for row in ret_arr:
        mystr += str(row) + "\n"

    return Response(mystr, mimetype='text/plain')

@app.route('/getMessagesByKeywordSubject/<word>')
def getMessagesByKeywordSubject(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = dataConn.getMessagesByKeywordSubject(word)
    mystr = ""
    for row in ret_arr:
        mystr += str(row) + "\n"

    return Response(mystr, mimetype='text/plain')

@app.route('/getMessageRecipientsByMessage/<message>')
def getMessageRecipientsByMessage(message):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = []
    ret_arr += dataConn.getRecipientsByMessage(message)

    mystr = ""
    for row in ret_arr:
        mystr += str(row) + "\n"

    return Response(mystr, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=False)
