#!/usr/bin/python
from flask import Flask
from flask import Response
import mysql_connect as mysql_conn
import activeSearchInterface as asI
##
# To run this, make sure the permissions are right:
# chmod a+x daemon_service.py 
#
# Then run it:
# ./daemon_service.py
##

app = Flask(__name__)

db = mysql_conn.mysql_connect("scottwalker")
activeSearch = asI.kernelAS()
wMat = mysql_conn.getFinalFeatureMatrix(db, 0, 0)
activeSearch.initialize(wMat)

# track the message ID that we're currently presenting the user for evaluation
currentMessage = -1

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

## functions that we had implemented in the Java version for TK's version of active search
@app.route('/firstmessage/<message>')
def firstMessage(message):
    activeSearch.firstMessage(int(message))
    return Response("hello",  mimetype='text/plain')

@app.route('/messageinteresting')
def interestingMessage():
    res = activeSearch.interestingMessage()
    return Response(res,  mimetype='text/plain')

@app.route('/messageboring')
def boringMessage():
    res = activeSearch.boringMessage()
    return Response(res,  mimetype='text/plain')

@app.route('/setalpha/<alpha>')
def setalpha(alpha):
    activeSearch.setalpha(double(alpha))
    return Response("hello",  mimetype='text/plain')

@app.route('/getStartPoint')
def getStartPoint():
    res = activeSearch.getStartPoint()
    return Response(res,  mimetype='text/plain')

@app.route('/resetLabel/<index>/<value>')
def resetLabel(index, value):
    activeSearch.resetLabel(int(index), int(value))
    return Response("hello",  mimetype='text/plain')

@app.route('/setLabelCurrent/<value>')
def setLabelCurrent(value):
    activeSearch.setLabelCurrent(int(value))
    return Response("hello",  mimetype='text/plain')

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
    return Response(res,  mimetype='text/plain')

@app.route('/pickRandomLabeledMessage')
def pickRandomLabeledMessage():
    res = activeSearch.pickRandomLabeledMessage()
    return Response(res,  mimetype='text/plain')

@app.route('/getLabel/<message>')
def getLabel(message):
    res = activeSearch.getLabel(int(message))
    return Response(res,  mimetype='text/plain')

#####
# For documentation on the following functions, see their analogs in mysql_connect.py
#####

@app.route('/getUserNameFromID/<id>')
def getUserNameFromID(id):
    return Response(mysql_conn.getUserNameFromID(int(id), db), mimetype='text/plain')

@app.route('/getMessagesFromUserToUser/<from_id>/<to_id>')
def getMessagesFromUserToUser(from_id, to_id):
    return Response(mysql_conn.getMessagesFromUserToUser(int(from_id), int(to_id), db), mimetype='text/plain')

@app.route('/getMessageSubjectFromMessageID/<id>')
def getMessageSubjectFromMessageID(id):
    return Response(mysql_conn.getMessageSubjectFromMessageID(int(id), db), mimetype='text/plain')

@app.route('/getMessageBodyFromMessageID/<id>')
def getMessageBodyFromMessageID(id):
    return Response(mysql_conn.getMessageBodyFromMessageID(int(id), db), mimetype='text/plain')

@app.route('/getTotalMessageCount')
def getTotalMessageCount():
    return Response(mysql_conn.getTotalMessageCount(db), mimetype='text/plain')

@app.route('/getMessageTimesAndSenders/<id>')
def getMessageTimesAndSenders(id):
    return Response(mysql_conn.getMessageTimesAndSenders(db), mimetype='text/plain')

@app.route('/getUsersByMessage/<id>')
def getUsersByMessage(id):
    return Response(mysql_conn.getUsersByMessage(int(id), db), mimetype='text/plain')

@app.route('/getSenderByMessage/<id>')
def getSenderByMessage(id):
    return Response(str(mysql_conn.getSenderByMessage(int(id), db)), mimetype='text/plain')

@app.route('/getTimeByMessage/<id>')
def getTimeByMessage(id):
    return Response(mysql_conn.getTimeByMessage(int(id), db), mimetype='text/plain')

@app.route('/getSubjectByMessage/<id>')
def getSubjectByMessage(id):
    return Response(mysql_conn.getSubjectByMessage(int(id), db), mimetype='text/plain')

@app.route('/getMessagesByKeyword/<word>')
def getMessagesByKeyword(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getMessagesByKeyword(word, db)
    mystr = ""
    for row in ret_arr:
        mystr += str(row) + "\n"

    return Response(mystr, mimetype='text/plain')

@app.route('/getMessagesByKeywordSubject/<word>')
def getMessagesByKeywordSubject(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getMessagesByKeywordSubject(word, db)
    mystr = ""
    for row in ret_arr:
        mystr += str(row) + "\n"

    return Response(mystr, mimetype='text/plain')

@app.route('/getMessageRecipientsByMessage/<message>')
def getMessageRecipientsByMessage(message):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getRecipientsByMessage(message, db)

    mystr = ""
    for row in ret_arr:
        mystr += str(row) + "\n"

    return Response(mystr, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
