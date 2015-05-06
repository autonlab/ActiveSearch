#!/usr/bin/python
from flask import Flask
from flask import Response
import mysql_connect as mysql_conn

app = Flask(__name__)

db = mysql_conn.mysql_connect("scottwalker")

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

## functions that we had implemented in the Java version for TK's version of active search
#@app.route('/firstemail/<email>/<mode>')
#@app.route('/firstemail/<email>')
#@app.route('/emailinteresting')
#@app.route('/emailboring')
#@app.route('/setalpha/<alpha>')
#@app.route('/getStartPoint')
#@app.route('/resetLabel/<index>/<value>')
#@app.route('/setLabelCurrent/<value>')
#@app.route('/setLabel/<index>/<value>')
#@app.route('/setLabelBulk/<csv>') # input is [index, value [,index, value etc]]
#@app.route('/getNextEmail')
#@app.route('/pickRandomLabeledEmail')
#@app.route('/getLabel/<email>')

#####
# For documentation on these functions, see their analogs in mysql_connect.py
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
