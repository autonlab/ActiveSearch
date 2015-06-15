package org.autonlab.activesearch;

import org.autonlab.activesearch.DataConnectionMySQL;
import org.autonlab.activesearch.SparseMatrix;
import org.autonlab.activesearch.ParallelSparseMatrixMultiply;

import org.apache.commons.math3.linear.*;

import org.apache.commons.math3.linear.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import org.jblas.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import java.util.Scanner;

import no.uib.cipr.matrix.sparse.*;
import no.uib.cipr.matrix.*;

public class EmailSimilarity implements Runnable {
    int myThreadID;
    int matrixType;

    static int threadCount;
    static SparseMatrix similarityMatrixSparse;

    static DoubleMatrix similarityMatrixReal;
    static DoubleMatrix emailWordSimilarity;
    static DoubleMatrix edgeSimilarityMatrix;

    static DataConnectionMySQL dataConnection;
    static int emailCount;
    static int[] emailTime;
    static int[] emailSender;
    static ArrayList<LinkedList<Integer>> emailRecipients;
    static Thread[] threadArray;

    int doWrites = 1;

    /**
     * @param matrixType 0 = RealMatrix (standard) 1 = SparseMatrix (used to write a sparse matrix to a file)
     * We used a SparseMatrix to verify that the data matched what
     * we were generating with Matlab
     * @param myThreadId this thread's index of all of the threads starting from zero
     */
    public EmailSimilarity(int thisThreadID, int type) {
	matrixType = type;
	myThreadID = thisThreadID;
    }

    /* to free up memory when this processing step is done */
    public static void clearStaticValues() {
	threadCount = 0;
	similarityMatrixSparse = null;
	similarityMatrixReal = null;
	emailWordSimilarity = null;
	edgeSimilarityMatrix = null;
	dataConnection = null;
	emailCount = 0;
	emailTime = null;
	emailSender = null;
	emailRecipients = null;
	threadArray = null;
    }

    public void dontDoWrites() {
	doWrites = 0;
    }

    /**
     * @param threadCount the total number of threads
     *
     * Only call this function once (say, just on thread 0). It initializes static data
     */
    public void initSimilarityData(int thisThreadCount) {
	int i;
	int j;
	threadCount = thisThreadCount;

	dataConnection = new DataConnectionMySQL();
	emailCount = dataConnection.getTotalEmailCount();

	if (matrixType == 0) {
	    //	    similarityMatrixReal = new Array2DRowRealMatrix(emailCount, emailCount);
	}
	else if (matrixType == 1) {
	    similarityMatrixSparse = new SparseMatrix(emailCount);
	}


	System.out.println("Loading email times and senders into memory");
	emailTime = new int[emailCount];
	emailSender = new int[emailCount];
	String[] messages = dataConnection.getEmailTimesAndSenders().split("[\\r\\n]+");
	if (messages.length != emailCount) {
	    throw new RuntimeException("Error email count " + emailCount + " and messages length " + messages.length + " mismatched");
	}
	for (i = 0; i < messages.length; i++) {
	    // format: messageID messageTime senderID
	    String[] retColumns = messages[i].split(" ");
	    emailTime[Integer.parseInt(retColumns[0])] = Integer.parseInt(retColumns[1]);
	    emailSender[Integer.parseInt(retColumns[0])] = Integer.parseInt(retColumns[2]);
	}
	messages = null;


	System.out.println("Loading email recipients into memory");
	emailRecipients = new ArrayList<LinkedList<Integer>>(emailCount);
	for (i = 0; i < emailCount; i++) {
	    emailRecipients.add(null);
	    String[] recipientList = dataConnection.getEmailRecipientsByEmail(i).split("[\\r\\n]+");
	    LinkedList<Integer> tempList = new LinkedList<Integer>();
	    for (j = 0; j < recipientList.length; j++) {
		if (!recipientList[j].isEmpty()) {
		    tempList.add(Integer.parseInt(recipientList[j]));
		}
	    }
	    emailRecipients.add(i, tempList);
	}

	System.out.println("Loading word frequency matrix");
	DoubleMatrix emailWordFrequency = dataConnection.getTFIDFSimilarity();

	/**
	 ** for sibi
	 */
	System.out.println("rows " + emailWordFrequency.rows + " cols" + emailWordFrequency.columns);
	SparseMatrix dump_matrix_for_sibi = new SparseMatrix(Math.max(emailWordFrequency.columns, emailWordFrequency.rows));
	for(i=0;i<emailWordFrequency.rows;i++){
	    for(j=0;j<emailWordFrequency.columns;j++) {
		if (emailWordFrequency.get(i,j) != 0.0) {
		    dump_matrix_for_sibi.put(i,j,emailWordFrequency.get(i,j));
		}
	    }
	}
	
	dump_matrix_for_sibi.write("sibi.txt");
	dump_matrix_for_sibi = null;
	/**
	 ** for sibi
	 */

	System.out.println("Calculating TFIDF matrix");

	DoubleMatrix s = MatrixFunctions.sqrt(MatrixFunctions.pow(emailWordFrequency,2).rowSums());
	emailWordFrequency.diviColumnVector(s);	

	/**
	 ** for sibi
	 */
	System.out.println("rows " + emailWordFrequency.rows + " cols" + emailWordFrequency.columns);
	dump_matrix_for_sibi = new SparseMatrix(Math.max(emailWordFrequency.columns, emailWordFrequency.rows));
	for(i=0;i<emailWordFrequency.rows;i++){
	    for(j=0;j<emailWordFrequency.columns;j++) {
		if (emailWordFrequency.get(i,j) != 0.0) {
		    dump_matrix_for_sibi.put(i,j,emailWordFrequency.get(i,j));
		}
	    }
	}
	
	dump_matrix_for_sibi.write("sibi2.txt");
	dump_matrix_for_sibi = null;
	if (1==1){
	    return;
	}
	/**
	 ** for sibi
	 */

	System.out.println("Loading up sparses");
	
	/*
	 * The word similarity matrix is very sparse (i.e., most emails contain very few of all possible words). In the Scott Walker
	 * dataset that we used, the matrix had only 0.031% nonzero values. Despite copying the matrices to a different math library
	 * it runs much faster and uses less memory
	 *
	 * Even as the number of emails scales, we believe that the sparsity of the matrices will stay relatively constant. Therefore
	 * the more emails there are, the more it makes sense to use a sparse library to perform the calculation.
	 */
	ParallelSparseMatrixMultiply[] sparseMatrices = new ParallelSparseMatrixMultiply[threadCount];
	for (i = 0; i < threadCount; i++) {
	    if (i == 0) {
		sparseMatrices[i] = new ParallelSparseMatrixMultiply(i, threadCount, emailWordFrequency);
	    }
	    else {
		sparseMatrices[i] = new ParallelSparseMatrixMultiply(i);
	    }
	    sparseMatrices[i].multiply();
	}
	emailWordSimilarity = ParallelSparseMatrixMultiply.getResult();
	System.out.println("Done loading up sparses");
	for (i = 0; i < threadCount; i++) {
	    sparseMatrices[i] = null;
	}
	ParallelSparseMatrixMultiply.clearStaticValues();

	if (emailWordSimilarity.rows != emailWordSimilarity.columns) {
	    throw new RuntimeException("email word similarity was not a square");
	}
	//	GlapEigenmap.write(emailWordSimilarity, "foo.out", emailWordSimilarity.rows, emailWordSimilarity.columns);

	System.out.println("Done loading data");

	edgeSimilarityMatrix = new DoubleMatrix(emailWordSimilarity.rows, emailWordSimilarity.columns);
	threadArray = new Thread[threadCount];
    }

    /**
     * Calculate the similarity between one message and another
     *
     * @in i message ID to consider
     * @in j message ID to compare to
     *
     * @return a similarity from 0 to 1 inclusive where 1 is perfect similarity and 0 is no similarity
     */
    public double getSimilarityOne(int i, int j) {
	double timeSimilarity = 0.0;
	double edgeSimilarity = 0.0;
	double similarity = 0.0;

	// if i==j we know the similarity is 1. We can either skip some math
	// or verify that this code produces the expected result
	int size1 = 0;
	if (emailRecipients.get(i) != null) {
	    size1 = emailRecipients.get(i).size();
	}

	int size2 = 0;
	LinkedList<Integer> temp = emailRecipients.get(j);
	if (temp != null) {
	    size2 = temp.size();
	}
	int index1 = 0;
	int index2 = 0;
	int commonCount = 0;

	// walk the list of recipients and count common values
	while (index1 < size1 && index2 < size2) {
	    if (emailRecipients.get(i).get(index1).intValue() == emailRecipients.get(j).get(index2).intValue()) {
		commonCount++;
		index1++;
		index2++; // user IDs can't be duplicated so we can increment both indices
	    } else if (emailRecipients.get(i).get(index1).intValue() < emailRecipients.get(j).get(index2).intValue()) {
		    index1++;
	    } else {
		index2++;
	    }
	}

	// include the sender in the calculation
	// If sender is not in its own recipent list but is in the other, count the commonality
	if (!(emailRecipients.get(i).contains(new Integer(emailSender[i]))) && 
	    (emailRecipients.get(j).contains(new Integer(emailSender[i])))) {
	    commonCount++;
	}
	if (!(emailRecipients.get(j).contains(new Integer(emailSender[j]))) && 
	    (emailRecipients.get(i).contains(new Integer(emailSender[j])))) {
	    commonCount++;
	}
	// If senders are not in their own recipient lists and are equal to each other, that's a commonality
	if (!(emailRecipients.get(i).contains(new Integer(emailSender[i]))) &&
	    !(emailRecipients.get(j).contains(new Integer(emailSender[j]))) &&
	    emailSender[i] == emailSender[j]) {
	    commonCount++;
	}
	// If the sender is not in the recipient list, increase the count
	if (!(emailRecipients.get(i).contains(new Integer(emailSender[i])))) {
	    size1++;
	}
	if (!(emailRecipients.get(j).contains(new Integer(emailSender[j])))) {
	    size2++;
	}

	// 13168189440000 is 6 weeks in seconds then squared
	//	timeSimilarity = Math.exp(-1.0 * (Math.pow(emailTime[i]-emailTime[j],2) / 52672757760000.0));
	timeSimilarity = Math.exp(-1.0 * (Math.pow(emailTime[i]-emailTime[j],2) / 13168189440000.0));
	//emailWordSimilarity.put(i,j,emailWordSimilarity.get(i,j) * timeSimilarity);

	// if the edge similarity is going to be zero we can skip the rest of the calculations

	if (size1 > 0 && size2 > 0 && commonCount > 0) {
	    edgeSimilarity = ((double)commonCount) / (Math.sqrt(size1*size2));

	    // 7257600 is 12 weeks in seconds
	    // 52672757760000 is that squared

	    //timeSimilarity = Math.exp(-1.0 * (Math.pow(emailTime[i]-emailTime[j],2) / 52672757760000.0));
	     //similarity = timeSimilarity * ((edgeSimilarity + emailWordSimilarity.get(i,j)) / 2.0);

	    // we wanted to split similarity into two files: 
	    //similarity = timeSimilarity * edgeSimilarity;
	}

	similarity = timeSimilarity * ((edgeSimilarity*.2) + (emailWordSimilarity.get(i,j) * .8));
	if(i==j && (similarity < 0.999 || similarity > 1.0001)) {
	    System.out.println("Email " + i + " similarity with itself was " + similarity + " instead of 1.0! Resetting to 1.0");
	    return 1.0;
	}
	edgeSimilarityMatrix.put(i,j,edgeSimilarity);
	return similarity;
    }

	    
    /**
     * Start building similarity matrix for this thread's subset of the work
     */
    public void buildMatrix() 
    {
	threadArray[myThreadID] = new Thread(this);
	threadArray[myThreadID].start();	
    }
    /**
     * Only call this function once (say, just on thread 0).  This
     * waits for the threads to finish, writes the sparse matrix to a file, then runs sanity checks
     */
    public void waitThreads()
    {
	int i;
	int j;
	for (i = 0; i < threadCount; i++) {
	    try {
		threadArray[i].join();
	    } catch (InterruptedException e) {
		e.printStackTrace();
	    }
	}
	if (matrixType != 1) {
	    return;
	}

	// always write this because it's an input to SearchMain
	System.out.println("Writing similarity row sum file");
	DoubleMatrix ARowsums = similarityMatrixSparse.rowSums();
	GlapEigenmap.write(ARowsums, ActiveSearchConstants.SIMILARITY_MATRIX_FILE, ARowsums.length, 1);

	/* these files are big and not always needed */
	if (doWrites == 1) {
	    System.out.println("Writing combo file");
	    similarityMatrixSparse.write("similarity_" + emailCount + "_combo.out");
	    System.out.println("Writing word file");
	    SparseMatrix.writeDoubleMatrix(emailWordSimilarity, "similarity_" + emailCount + "_word");
	    System.out.println("Writing edge file");
	    SparseMatrix.writeDoubleMatrix(edgeSimilarityMatrix, "similarity_" + emailCount + "_edge");
	}

	// sanity checking unit tests
	// Make sure every email's similiarity with itself is 1
	for (i = 0; i < emailCount; i++) {
	    if (similarityMatrixSparse.get(i,i) > 1.000001 ||
		similarityMatrixSparse.get(i,i) < .9999999) {
		System.out.println("ERROR! similarity between " + i + " and itself isn't close to 1, it's " + similarityMatrixSparse.get(i,i));
		throw new RuntimeException();
	    }
	}
	System.out.println("Identity check complete");

	// Make sure no email has a similarity higher than 1 or < 0
	int count = 0;
	for (i = 0; i < emailCount; i++) {
	    for (j = i+1; j < emailCount; j++) {
		double similarityValue = similarityMatrixSparse.get(i,j);
		double similarityValueInverse = similarityMatrixSparse.get(j,i);
		if (similarityValue > 1.000001 || similarityValue < -0.000001) {
		    // Expect that no emails have similarity 1 unless it's with itself
		    System.out.println("ERROR! similarity between " + i + " and " + j + " is unexpectedly >1 : " + similarityValue);
		    throw new RuntimeException();
		}
		if (similarityValue > (similarityValueInverse + .000001) ||
		    similarityValue < (similarityValueInverse - .000001)) {
		    // i,j should equal j,i
		    System.out.println("ERROR! similarity between " + i + " and " + j + " doesn't match the reverse! " + similarityValue + " and " + similarityValueInverse);
		    throw new RuntimeException();
		}
	    }
	}
	System.out.println("Values check complete");
    }

    public void run() {
	int i;
	int j;
	double similarity;
	int workIndex = emailCount/threadCount;
	if ((emailCount % threadCount) != 0) { 	// round up so the last thread doesn't drop rows
	    workIndex++;
	}
	int startID = workIndex * myThreadID;
	int endID = Math.min(workIndex * (myThreadID+1), emailCount);

	System.out.println("Thread " + myThreadID + " started from " + startID + " to (but not including)" + endID);

	for (i = startID; i < endID; i++) {
	    if ((i % 500) == 0) {
		System.out.println("Thread " + myThreadID + ": " + (i-startID) + " / " + (endID-startID));
	    }
	    for (j=0;j<emailCount;j++) {
		similarity = this.getSimilarityOne(i,j);
		if (matrixType == 0) {
		    //similarityMatrixReal.setEntry(i, j, similarity);
		}
		else if (matrixType == 1) {
		    similarityMatrixSparse.put(i, j, similarity);
		}
	    }
	}
	System.out.println("Thread " + myThreadID + ": thread done");
    }

    public DoubleMatrix getMatrixFile(){
	int i;
	int j;
	if (matrixType == 1) {
	    DoubleMatrix retMatrix = new DoubleMatrix(emailCount, emailCount);
	    for (i = 0; i < emailCount; i++) {
		for (j = 0; j < emailCount; j++) {
		    retMatrix.put(i, j, similarityMatrixSparse.get(i, j));
		}
	    }
	    return retMatrix;
	}
	return similarityMatrixReal;
    }

    /* this is not an efficient way of converting our sparse matrix to the mtj sparse matrix
       however using mtj is a proof of concept. We might replace our sparse implementation
       with the mtj one eventually *//*
    public LinkedSparseMatrix getLinkedSparseMatrix() {
	int i;
	int j;
	LinkedSparseMatrix retMatrix = new LinkedSparseMatrix(emailCount, emailCount);
	for (i = 0; i < emailCount; i++) {
	    for (j = 0; j < emailCount; j++) {
		// LinkedSparseMatrix knows not to store zeros
		retMatrix.set(i, j, similarityMatrixSparse.get(i, j));
	    }
	}
	return retMatrix;

    }
				     */
    public DoubleMatrix readMatrixFile(String filename) {
	BufferedReader br = null;
	emailCount = 0;
	
	int count = 0;
	try {
 
	    String sCurrentLine;
	    String[] sParts;
	    br = new BufferedReader(new FileReader(filename));
	    System.out.print("Importing matrix ");
	    while ((sCurrentLine = br.readLine()) != null) {
		sParts = sCurrentLine.split(" ");
		if (emailCount == 0) {
		    emailCount = sParts.length;
		    similarityMatrixReal = new DoubleMatrix(emailCount, emailCount);
		}

		if (count % 200000 == 0) {
		    //System.out.println(count + ":" + Integer.parseInt(sParts[0]) + " " + Integer.parseInt(sParts[1]) + " " + Double.parseDouble(sParts[2]));
		    System.out.print(".");
		}

		int myX = Integer.parseInt(sParts[0]);
		int myY = Integer.parseInt(sParts[1]);
		// the similarity file is written in matlab format where indices start at 1 not 0
		myX--;
		myY--;
		double myVal = Double.parseDouble(sParts[2]);
		similarityMatrixReal.put(myX, myY, myVal);	
		if (myX != myY) {
		    similarityMatrixReal.put(myY, myX, myVal);
		}
		count++;
	    }
	    System.out.println("");
	} catch (IOException e) {
	    e.printStackTrace();
	} finally {
	    try {
		if (br != null)br.close();
	    } catch (IOException ex) {
		ex.printStackTrace();
	    }
	}
	return similarityMatrixReal;
    }
}