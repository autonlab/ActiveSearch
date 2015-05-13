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

# track the email ID that we're currently presenting the user for evaluation
currentEmail = -1

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

## functions that we had implemented in the Java version for TK's version of active search
@app.route('/firstemail/<email>')
def firstEmail(email):
    activeSearch.firstEmail(email)
    return Response("hello",  mimetype='text/plain')

@app.route('/emailinteresting')
def interestingEmail():
    res = activeSearch.interestingEmail()
    return Response(res,  mimetype='text/plain')

@app.route('/emailboring')
def boringEmail():
    res = activeSearch.boringEmail()
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

@app.route('/getNextEmail')
def getNextEmail():
    res = activeSearch.getNextEmail()
    return Response(res,  mimetype='text/plain')

@app.route('/pickRandomLabeledEmail')
def pickRandomLabeledEmail():
    res = activeSearch.pickRandomLabeledEmail()
    return Response(res,  mimetype='text/plain')

@app.route('/getLabel/<email>')
def getLabel(email):
    res = activeSearch.getLabel(email)
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

@app.route('/getEmailSubjectFromMessageID/<id>')
def getEmailSubjectFromMessageID(id):
    return Response(mysql_conn.getMessageSubjectFromMessageID(id, db), mimetype='text/plain')

@app.route('/getEmailBodyFromMessageID/<id>')
def getEmailBodyFromMessageID(id):
    return Response(mysql_conn.getMessageBodyFromMessageID(id, db), mimetype='text/plain')

@app.route('/getTotalEmailCount')
def getTotalEmailCount():
    return Response(mysql_conn.getTotalMessageCount(db), mimetype='text/plain')

@app.route('/getEmailTimesAndSenders/<id>')
def getEmailTimesAndSenders(id):
    return Response(mysql_conn.getMessageTimesAndSenders(db), mimetype='text/plain')

@app.route('/getUsersByEmail/<id>')
def getUsersByEmail(id):
    return Response(mysql_conn.getUsersByMessage(id, db), mimetype='text/plain')

@app.route('/getSenderByEmail/<id>')
def getSenderByEmail(id):
    return Response(mysql_conn.getSenderByMessage(id, db), mimetype='text/plain')

@app.route('/getTimeByEmail/<id>')
def getTimeByEmail(id):
    return Response(mysql_conn.getTimeByMessage(id, db), mimetype='text/plain')

@app.route('/getSubjectByEmail/<id>')
def getSubjectByEmail(id):
    return Response(mysql_conn.getSubjectByMessage(id, db), mimetype='text/plain')

@app.route('/getEmailsByKeyword/<word>')
def getEmailsByKeyword(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getMessagesByKeyword(word, db)
    str = ""
    for row in ret_arr:
        str += row + "\n"

    return Response(str, mimetype='text/plain')

@app.route('/getEmailsByKeywordSubject/<word>')
def getEmailsByKeywordSubject(word):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getMessagesByKeywordSubject(word, db)
    str = ""
    for row in ret_arr:
        str += row + "\n"

    return Response(str, mimetype='text/plain')

@app.route('/getEmailRecipientsByEmail/<email>')
def getEmailRecipientsByEmail(email):
    # this returns an array of entries so we have to concatenate them into a big string
    ret_arr = mysql_conn.getRecipientsByMessage(email, db)

    str = ""
    for row in ret_arr:
        str += row + "\n"

    return Response(str, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
