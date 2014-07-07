package org.autonlab.activesearch;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.ListIterator;

import org.jblas.*;
import org.autonlab.activesearch.ActiveSearchConstants;

public class DataConnectionMySQL {

    Connection con = null;
    int[] emailSender;

    /**
     * Create a new DataConnectionMySQL object. This establishes a connection to the mySQL database
     */
    public DataConnectionMySQL()
    {
	try {
	    // The documentation says this is deprecated but I can't get this app to work without it
	    Class.forName("com.mysql.jdbc.Driver");
	}
	catch (ClassNotFoundException ex) {
	    System.out.println(ex.getMessage());
	}

	try {
	     con = DriverManager.getConnection("jdbc:mysql://localhost:" + ActiveSearchConstants.DATABASE_PORT + "/" + ActiveSearchConstants.DATABASE_NAME + "?zeroDateTimeBehavior=convertToNull&mysql_enable_utf8=1", ActiveSearchConstants.DATABASE_USERNAME, ActiveSearchConstants.DATABASE_PASSWORD);
	} catch (SQLException ex) {
	    System.out.println(ex.getMessage());
	}
    }

    /**
     * Translate from a user ID to a username string
     *
     * @param userID The user ID to look up
     *
     * @return the username
     */
    public String getUserNameFromID(int userID)
    {
	PreparedStatement pst = null;
	ResultSet rs = null;
	String userName = "";

	try {
	    pst = con.prepareStatement("SELECT * FROM people WHERE personid=" + userID);
	    rs = pst.executeQuery();
	    rs.next();

	    // use name if possible, fall back to email address if names aren't listed	
	    userName = rs.getString(3);
	    if (userName == null || userName.length() == 0) {	
		userName = rs.getString(2);
	    }
	} catch (SQLException ex) {
	    System.out.println("error getting user name via mySQL for user id " + userID + ex.getMessage());
	    ex.printStackTrace();
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("error cleaning up after getting user name via mySQL for user id " + userID + ex.getMessage());
		ex.printStackTrace();
	    }
	}
	return userName;
    }

    /**
     * Get a linked list of message subjects sent from one user ID to another user ID
     *
     * @param userIDFrom User ID of the user who sent the emails
     * @param userIDTo User ID of the user who received the emails
     *
     * @return Linked list of email message subjects sorted by timestamp of the form <messageid> : <timestamp> : <subject>
     */
    public String getMessagesFromUserToUser(int userIDFrom, int userIDTo)
    {
	String emailList = new String();
	PreparedStatement pst = null;
	ResultSet rs = null;

	try {
	    pst = con.prepareStatement("SELECT messages.messageid, messages.messagedt, messages.subject"
				       + " FROM messages, recipients WHERE messages.messageid=recipients.messageid AND recipients.personid="
				       + userIDTo + " AND messages.senderid=" + userIDFrom 
				       + " GROUP BY messages.messageid ORDER BY messages.messagedt");

	    rs = pst.executeQuery();
	    while (rs.next()) {
		emailList += rs.getString(1) + " : " + rs.getString(2) + " : " + rs.getString(3) + "\n";
	    }
	} catch (SQLException ex) {
	    System.out.println("error getting messages via mySQL for user ids " + userIDFrom 
			       + " and " + userIDTo + ex.getMessage());
	    ex.printStackTrace();
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("error cleaning up getting messages via mySQL for user ids " + userIDFrom 
				   + " and " + userIDTo + ex.getMessage());
	    }
	}
	return emailList;
    }

    /**
     * Get a message subject from a message ID
     *
     * @param messageID ID of the message to retrieve
     *
     * @return Message subject
     */
    public String getEmailSubjectFromMessageID(int messageID)
    {
	String emailSubject = "";
	PreparedStatement pst = null;
	ResultSet rs = null;
	try {
	    pst = con.prepareStatement("SELECT messages.subject FROM messages WHERE messages.messageid=" + messageID);
	    rs = pst.executeQuery();
	    rs.next();
	    emailSubject = rs.getString(1);
	} catch (SQLException ex) {
	    System.out.println("Exception looking up subject for messageID " + messageID + " : " + ex.getMessage());
	    throw new RuntimeException();
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception looking up subject for messageID " + messageID + " : " + ex.getMessage());
		throw new RuntimeException();
	    }
	}
	return emailSubject;
    }

    /**
     * Get a message body from a message ID
     *
     * @param messageID ID of the message to retrieve
     *
     * @return Message body contents with some other email info prepended to it
     */
    public String getEmailBodyFromMessageID(int messageID)
    {
	String emailText = "";
	PreparedStatement pst = null;
	ResultSet rs = null;
	try {
	    pst = con.prepareStatement("SELECT messages.messagedt, messages.subject, bodies.body FROM messages, bodies WHERE messages.messageid=bodies.messageid AND messages.messageid=" + messageID);
	    rs = pst.executeQuery();
	    rs.next();
	    emailText = rs.getString(1) + "\n\n" + rs.getString(2) + "\n\n" + rs.getString(3);
	} catch (SQLException ex) {
	    System.out.println("Exception looking up messageID " + messageID + " : " + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception looking up messageID " + messageID + " : " + ex.getMessage());
	    }
	}
	return emailText;
    }

    /**
     * Initialize total number of email messages. Call this before functions that use it
     *
     * @return total number of email messages
     */
    public int getTotalEmailCount()
    {
	int totalEmailCount = 0;
	PreparedStatement pst = null;
	ResultSet rs = null;
	try {
	    pst = con.prepareStatement("SELECT COUNT(messageID) FROM messages");
	    rs = pst.executeQuery();
	    rs.next();
	    totalEmailCount = rs.getInt(1);
	} catch (SQLException ex) {
	    System.out.println("Exception getting total email count" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting total email count" + ex.getMessage());
	    }
	}
	return totalEmailCount;
    }

    /**
     * Get arrays of email timestamps and senderIDs
     * The index of the array is the email message ID. The timestamp is seconds from epoch
     *
     * @return 2 item array with the first item being the array of timestamps and the second the array of senderIDs
     */
    public String getEmailTimesAndSenders()
    {
	String retVal = new String();
	PreparedStatement pst = null;
	ResultSet rs = null;
	try {
	    pst = con.prepareStatement("SELECT messageid, UNIX_TIMESTAMP(messagedt), senderid FROM messages");
	    rs = pst.executeQuery();
	    while (rs.next()) {
		retVal += rs.getInt(1) + " " + rs.getInt(2) + " " + rs.getInt(3) + "\n";
	    }
	} catch (SQLException ex) {
	    System.out.println("Exception getting email times and senders" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting email times and senders" + ex.getMessage());
	    }
	}
	return retVal;
    }

    public String getUsersByEmail(int emailID)
    {
	String peopleList = new String();
	peopleList = getEmailRecipientsByEmail(emailID);
	peopleList += getSenderByEmail(emailID);
	return peopleList;
    }

    public int getSenderByEmail(int email) {
	PreparedStatement pst = null;
	ResultSet rs = null;

	int ret = -1;
	try {
	    pst = con.prepareStatement("SELECT senderid FROM messages WHERE messageid = " + email + " LIMIT 1");
	    rs = pst.executeQuery();
	    rs.next();
	    ret = rs.getInt(1);
	} catch (SQLException ex) {
	    System.out.println("Exception getting sender by email" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting sender by email" + ex.getMessage());
	    }
	}
	return ret;
    }

    public String getTimeByEmail(int email) {
	PreparedStatement pst = null;
	ResultSet rs = null;
	String ret = "";

	try {
	    pst = con.prepareStatement("SELECT messagedt FROM messages WHERE messageid = " + email + " LIMIT 1");
	    rs = pst.executeQuery();
	    rs.next();
	    ret = rs.getString(1);
	} catch (SQLException ex) {
	    System.out.println("Exception getting time by email" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting time by email" + ex.getMessage());
	    }
	}
	return ret;
    }

    public String getSubjectByEmail(int email) {
	PreparedStatement pst = null;
	ResultSet rs = null;
	String ret = "";

	try {
	    pst = con.prepareStatement("SELECT subject FROM messages WHERE messageid = " + email + " LIMIT 1");
	    rs = pst.executeQuery();
	    rs.next();
	    ret = rs.getString(1);
	} catch (SQLException ex) {
	    System.out.println("Exception getting subject by email" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting subject by email" + ex.getMessage());
	    }
	}
	return ret;
    }



    // format: <messageid> : <timestamp> : <subject>
    public String getEmailsByKeyword(String word)
    {
	String retList = new String();
	PreparedStatement pst = null;
	ResultSet rs = null;

	try {
	    pst = con.prepareStatement("SELECT messages.messageid, messages.messagedt, messages.subject"
				       + " FROM messages INNER JOIN bodies ON messages.messageid=bodies.messageid "
				       + " WHERE body LIKE '%" + word + "%' ORDER BY messages.messagedt");
	    rs = pst.executeQuery();
	    while (rs.next()) {
	    	retList += rs.getString(1) + " : " + rs.getString(2) + " : " + rs.getString(3) + "\n";
	    }
	} catch (SQLException ex) {
	    System.out.println("Exception getting users emails by keyword" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting emails by keyword" + ex.getMessage());
	    }
	}

	return retList;
    }

    // format: <messageid> : <timestamp> : <subject>
    public String getEmailsByKeywordSubject(String word)
    {
	String retList = new String();
	PreparedStatement pst = null;
	ResultSet rs = null;

	try {
	    pst = con.prepareStatement("SELECT messages.messageid, messages.messagedt, messages.subject"
				       + " FROM messages WHERE subject LIKE '%" + word + "%' ORDER BY messagedt");
	    rs = pst.executeQuery();

	    while (rs.next()) {
		retList += rs.getString(1) + " : " + rs.getString(2) + " : " + rs.getString(3) + "\n";
	    }
	} catch (SQLException ex) {
	    System.out.println("Exception getting users emails by keyword" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting emails by keyword" + ex.getMessage());
	    }
	}

	return retList;
    }

    /**
     * Get an ArrayList of Integer Linked Lists. The array index is the messageID, and the linked list of
     * integers is the variable number of unique and sorted recipient user IDs for that message
     *
     * An ArrayList was used rather than a generic array due to compiler issues with the LinkedList<Integer> type
     *
     * @return ArrayList of LinkedList of recipient user IDs
     */
    /*
    public ArrayList<LinkedList<Integer>> getEmailRecipients()
    {
	int emailCount = this.getTotalEmailCount();
	ArrayList<LinkedList<Integer>> emailRecipients = new ArrayList<LinkedList<Integer>>(emailCount);

	int i;
	for (i = 0; i < emailCount; i++) {
	    emailRecipients.add(null);
	    LinkedList<Integer> tempList = getEmailRecipientsByEmail(i);
	    emailRecipients.add(i, tempList);
	}
	return emailRecipients;
	}*/
    
    public String getEmailRecipientsByEmail(int email)
    {
	PreparedStatement pst = null;
	ResultSet rs = null;

	String tempList = new String();

	try {
	    // use DISTINCT here because in the berkeley enron database personid can be repeated for to, cc, and bcc
	    pst = con.prepareStatement("SELECT DISTINCT personid FROM recipients WHERE messageid=" + email);
	    rs = pst.executeQuery();
	    while (rs.next()) {
		tempList += rs.getString(1) + "\n";
	    }
	} catch (SQLException ex) {
	    System.out.println("Exception getting email recipients " + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting email recipients" + ex.getMessage());
	    }
	}
	return tempList;
    }

    /**
     * Close a database connection that was opened by this class' constructor
     */
    public void closeConnection() {
	try {
	    if (con != null) {
		con.close();
	    }
	} catch (SQLException ex) {
	    System.out.println("Failed to close database connection");
	}
    }

    public int getTotalWordCount()
    {
	int totalWordCount = 0;
	PreparedStatement pst = null;
	ResultSet rs = null;
	try {
	    pst = con.prepareStatement("SELECT COUNT(word_id) FROM tf_idf_wordmap");
	    rs = pst.executeQuery();
	    rs.next();
	    totalWordCount = rs.getInt(1);
	} catch (SQLException ex) {
	    System.out.println("Exception getting total word count" + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting total word count" + ex.getMessage());
	    }
	}
	return totalWordCount;
    }

    /**
     * 
     */
    public DoubleMatrix getTFIDFSimilarity() {
	int numEmails = getTotalEmailCount();
	int numWords = getTotalWordCount();

	DoubleMatrix emailWordFrequency = new DoubleMatrix(numEmails, numWords);
	PreparedStatement pst = null;
	ResultSet rs = null;

	try {
	    pst = con.prepareStatement("SELECT * FROM tf_idf_dictionary ORDER BY messageid");
	    rs = pst.executeQuery();
	    while (rs.next()) {
		int wordID = rs.getInt(1);
		int currentMessageID = rs.getInt(2);
		int count = rs.getInt(3);
		emailWordFrequency.put(currentMessageID, wordID, count);
	    }
	} catch (SQLException ex) {
	    System.out.println("Exception getting word counts " + ex.getMessage());
	} finally {
	    try {
		if (pst != null) {
		    pst.close();
		}
	    } catch (SQLException ex) {
		System.out.println("Exception cleaning up after getting word counts" + ex.getMessage());
	    }
	}
	return emailWordFrequency;
    }

}
