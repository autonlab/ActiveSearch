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
activeSearch = asI.genericAS()

# track the message ID that we're currently presenting the user for evaluation
currentMessage = -1

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

## functions that we had implemented in the Java version for TK's version of active search
@app.route('/firstmessage/<message>')
def firstMessage(message):
    activeSearch.firstMessage(message)
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
    activeSearch.setalpha(alpha)
    return Response("hello",  mimetype='text/plain')

@app.route('/getStartPoint')
def getStartPoint():
    res = activeSearch.getStartPoint()
    return Response(res,  mimetype='text/plain')

@app.route('/resetLabel/<index>/<value>')
def resetLabel(index, value):
    activeSearch.resetLabel(index, value)
    return Response("hello",  mimetype='text/plain')

@app.route('/setLabelCurrent/<value>')
def setLabelCurrent(value):
    activeSearch.setLabelCurrent(value)
    return Response("hello",  mimetype='text/plain')

# input is [index, value [,index, value etc]]
@app.route('/setLabelBulk/<csv>')
def setLabeLBulk(csv):
    activeSearch.setLabelBulk(csv)
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
    res = activeSearch.getLabel(message)
    return Response(res,  mimetype='text/plain')

#####
# For documentation on the following functions, see their analogs in mysql_connect.py
#####

@app.route('/getUserNameFromID/<id>')
def getUserNameFromID(id):
    return Response(mysql_conn.getUserNameFromID(id, db), mimetype='text/plain')

@app.route('/getMessagesFromUserToUser/<from_id>/<to_id>')
def getMessagesFromUserToUser(from_id, to_id):
    return Response(mysql_conn.getMessagesFromUserToUser(from_id, to_id, db), mimetype='text/plain')

@app.route('/getMessageSubjectFromMessageID/<id>')
def getMessageSubjectFromMessageID(id):
    return Response(mysql_conn.getMessageSubjectFromMessageID(id, db), mimetype='text/plain')

@app.route('/getMessageBodyFromMessageID/<id>')
def getMessageBodyFromMessageID(id):
    return Response(mysql_conn.getMessageBodyFromMessageID(id, db), mimetype='text/plain')

@app.route('/getTotalMessageCount')
def getTotalMessageCount():
    return Response(mysql_conn.getTotalMessageCount(db), mimetype='text/plain')

@app.route('/getMessageTimesAndSenders/<id>')
def getMessageTimesAndSenders(id):
    return Response(mysql_conn.getMessageTimesAndSenders(db), mimetype='text/plain')

@app.route('/getUsersByMessage/<id>')
def getUsersByMessage(id):
    return Response(mysql_conn.getUsersByMessage(id, db), mimetype='text/plain')

@app.route('/getSenderByMessage/<id>')
def getSenderByMessage(id):
    return Response(mysql_conn.getSenderByMessage(id, db), mimetype='text/plain')

@app.route('/getTimeByMessage/<id>')
def getTimeByMessage(id):
    return Response(mysql_conn.getTimeByMessage(id, db), mimetype='text/plain')

@app.route('/getSubjectByMessage/<id>')
def getSubjectByMessage(id):
    return Response(mysql_conn.getSubjectByMessage(id, db), mimetype='text/plain')

@app.route('/getMessagesByKeyword/<word>')
def getMessagesByKeyword(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getMessagesByKeyword(word, db)
    str = ""
    for row in ret_arr:
        str += row + "\n"

    return Response(str, mimetype='text/plain')

@app.route('/getMessagesByKeywordSubject/<word>')
def getMessagesByKeywordSubject(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getMessagesByKeywordSubject(word, db)
    str = ""
    for row in ret_arr:
        str += row + "\n"

    return Response(str, mimetype='text/plain')

@app.route('/getMessageRecipientsByMessage/<message>')
def getMessageRecipientsByMessage(message):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getRecipientsByMessage(message, db)

    str = ""
    for row in ret_arr:
        str += row + "\n"

    return Response(str, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
