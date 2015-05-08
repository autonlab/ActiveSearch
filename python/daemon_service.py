#!/usr/bin/python
from flask import Flask
from flask import Response
import mysql_connect as mysql_conn

##
# To run this, make sure the permissions are right:
# chmod a+x daemon_service.py 
#
# Then run it:
# ./daemon_service.py
##

app = Flask(__name__)

db = mysql_conn.mysql_connect("scottwalker")

# track the email ID that we're currently presenting the user for evaluation
currentEmail = -1

#@app.route(...)
#def login():
#    username = request.args.get('username')
#    password = request.args.get('password')

## functions that we had implemented in the Java version for TK's version of active search
@app.route('/firstemail/<email>')
def firstEmail(email):
    #Initialize any datastructures (throw out old ones if they exist) and seed the algorithm with this email ID
    return Response("hello",  mimetype='text/plain')

@app.route('/emailinteresting')
def interestingEmail():
    # setLabel(currentEmail, 1)
    # res = getNextEmail()
    return Response("next email id",  mimetype='text/plain')

@app.route('/emailboring')
def boringEmail():
    # setLabel(currentEmail, 0)
    # res = getNextEmail()
    return Response("next email id",  mimetype='text/plain')

@app.route('/setalpha/<alpha>')
def setalpha(alpha):
    #set the tuning parameter alpha. I think we won't need this call anymore
    return Response("hello",  mimetype='text/plain')

@app.route('/getStartPoint')
def getStartPoint():
    # I think this just returned the first email that was used to seed this run
    return Response("hello",  mimetype='text/plain')

@app.route('/resetLabel/<index>/<value>')
def resetLabel(index, value):
    # set label of email <index> with <value>
    return Response("hello",  mimetype='text/plain')

@app.route('/setLabelCurrent/<value>')
def setLabelCurrent(value):
    # simply call setLabel(currentEmail, value)
    return Response("hello",  mimetype='text/plain')

@app.route('/setLabel/<index>/<value>')
def setLabel(index, value):
    # set the label for email <index> with value <value>
    return Response("hello",  mimetype='text/plain')

# input is [index, value [,index, value etc]]
@app.route('/setLabelBulk/<csv>')
def setLabeLBulk(csv):
    # loop over each pair in the csv and call setLabel on it
    # requested by Sotera
    return Response("hello",  mimetype='text/plain')

@app.route('/getNextEmail')
def getNextEmail():
    # call into active search and have it calculate the next email to show to the user
    # (this is usally called from boringEmail() or interestingEmail()
    return Response("<new email id>",  mimetype='text/plain')

@app.route('/pickRandomLabeledEmail')
def pickRandomLabeledEmail():
    #of the labeled emails, randomly return one
    # this is in case the user wants to pre-label many emails and we have to start from one of them
    return Response("<email id>",  mimetype='text/plain')

@app.route('/getLabel/<email>')
def getLabel(email):
    # return the label for a given email ID
    return Response("0/1",  mimetype='text/plain')

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
