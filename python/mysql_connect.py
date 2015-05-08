#!/usr/bin/python
import MySQLdb
import scipy.sparse as ss
import numpy.linalg as np

def mysql_connect(database, db_host="localhost", db_user="root", db_password=""):
    db = MySQLdb.connect(host=db_host,
                         user=db_user,
                         passwd=db_password,
                         db=database)
    return db

# retrieve the tf_idf frequencies from the database
# return a scipy.sparse.csr_matrix of the form [wordID][messageID] = count of times word seen in that message
def getTFIDFSimilarity(db):
    num_words = getTotalWordCount(db)
    num_messages = getTotalMessageCount(db)

    cur = db.cursor()
    cur.execute("SELECT * FROM tf_idf_dictionary ORDER BY messageid")

    ret_row = []
    ret_col = []
    ret_data = []

    for row in cur.fetchall():
        wordID = row[0]
        currentMessageID = row[1]
        count = row[2]
        ret_row.append(wordID)
        ret_col.append(currentMessageID)
        ret_data.append(count)

    ret_matrix = ss.csr_matrix((ret_data, (ret_row, ret_col)), shape=(num_words, num_messages))
    return ret_matrix
    
# return the number of words in the tf_idf computation
def getTotalWordCount(db):
    cur = db.cursor()
    cur.execute("SELECT COUNT(word_id) FROM tf_idf_wordmap")

    row=cur.fetchone()
    return row[0]

# return the number of messages
def getTotalMessageCount(db):
    cur = db.cursor()
    cur.execute("SELECT COUNT(messageID) FROM messages")

    row=cur.fetchone()
    return row[0]

# for a given message id, return an array of recipients of that message
def getRecipientsByMessage(message_id, db):
    cur = db.cursor()
    cur.execute("SELECT DISTINCT personid FROM recipients WHERE messageid=" + str(message_id))
    data = []

    for row in cur.fetchall():
        data.append(row[0])

    return data

# for a given a keyword, return an array of "<message_id> : <timestamp> : <subject>"
# for any message whose subject matches 
# we do not scrub the inputs here and assume the user won't do anything malicious
def getMessagesByKeywordSubject(word, db):
    cur = db.cursor()
    cur.execute("SELECT messages.messageid, messages.messagedt, messages.subject"
                + " FROM messages WHERE subject LIKE '%" + word + "%' ORDER BY messagedt")
    data = []

    for row in cur.fetchall():
        data.append(str(row[0]) + " : " + str(row[1]) + " : " + row[2])

    return data

# for a given a keyword, return an array of "<message_id> : <timestamp> : <subject>"
# for any message whose body
# we do not scrub the inputs here and assume the user won't do anything malicious
def getMessagesByKeyword(word, db):
    cur = db.cursor()
    cur.execute("SELECT messages.messageid, messages.messagedt, messages.subject"
                + " FROM messages INNER JOIN bodies ON messages.messageid=bodies.messageid "
                + " WHERE body LIKE '%" + word + "%' ORDER BY messages.messagedt")
    data = []

    for row in cur.fetchall():
        data.append(str(row[0]) + " : " + str(row[1]) + " : " + row[2])

    return data

# for a message_id, return the subject string
def getSubjectByMessage(message_id, db):
    return getFieldByMessage(message_id, "subject", db)

# for a message_id, return the timestamp
def getTimeByMessage(message_id, db):
    return getFieldByMessage(message_id, "messagedt", db)

# for a message_id, return the sender_id
def getSenderByMessage(message_id, db):
    return getFieldByMessage(message_id, "senderid", db)

# for a message_id, return a specific field
def getFieldByMessage(message_id, field_name, db):
    cur = db.cursor()
    cur.execute("SELECT " + field_name + " FROM messages WHERE messageid = " + str(message_id) + " LIMIT 1")

    row=cur.fetchone()
    return row[0]

# return an array listing all people involved in a message: the sender and the recipients
def getUsersByMessage(message_id):
    ret_array = getRecipientsByMessage(message_id, db)
    ret_array.append(getSenderByMessage(message_id, db))
    return ret_array

# returns an array where each value is a string of the form "<message_id> <seconds from epoch timestamp> <sender_id>"
def getEmailTimesAndSenders(db):
    cur = db.cursor()
    cur.execute("SELECT messageid, UNIX_TIMESTAMP(messagedt), senderid FROM messages")
    data = []

    for row in cur.fetchall():
        data.append(str(row[0]) + " " + str(row[1]) + " " + row[2])

    return data

# returns an array where the first element is the message timestamp, the second is the subject, and the third is the body
def getMessageBodyFromMessageID(message_id, db):
    cur = db.cursor()
    cur.execute("SELECT messages.messagedt, messages.subject, bodies.body FROM messages, bodies"
                + " WHERE messages.messageid=bodies.messageid AND messages.messageid=" + str(message_id))

    row=cur.fetchone()
    data = []
    data.append(row[0])
    data.append(row[1])
    data.append(row[2])

    return data

# returns a the subject string for a message_id
def getMessageSubjectFromMessageID(message_id, db):
    cur = db.cursor()
    cur.execute("SELECT messages.subject FROM messages WHERE messages.messageid=" + str(message_id))

    row=cur.fetchone()
    return row[0]

# for user_ids user_from and user_to, return an array where each value is a message that user_from sent and user_to received
# Each value is of the form "<message_id> : <timestamp> : <subject>"
def getMessagesFromUserToUser(user_from, user_to, db):
    cur = db.cursor()
    cur.execute("SELECT messages.messageid, messages.messagedt, messages.subject"
                + " FROM messages, recipients WHERE messages.messageid=recipients.messageid AND recipients.personid="
                + str(user_to) + " AND messages.senderid=" + str(user_from)
                + " GROUP BY messages.messageid ORDER BY messages.messagedt")
    data = []

    for row in cur.fetchall():
        data.append(str(row[0]) + " : " + str(row[1]) + " : " + row[2])

    return data

#returns the username string from a user_id
#If no name is available, return the email address instead
def getUserNameFromID(user_id, db):
    cur = db.cursor()
    cur.execute("SELECT * FROM people WHERE personid=" + str(user_id))

    row=cur.fetchone()

    email_address = row[1]
    name = row[2]

    if (name is None or len(name) == 0):
        return email_address

    return name


def getMatrix(db):
    similarity_data = getTFIDFSimilarity(db)
    s = ((similaritydata**2).sum(1))**0.5
    similarity_data = np.divide(similarity_data, s[:,np.newaxis])
    return similarity_data
