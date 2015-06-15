package org.autonlab.activesearch;

import org.autonlab.activesearch.DataConnectionMySQL;
import org.autonlab.activesearch.EmailSimilarity;
import org.autonlab.activesearch.SearchMain;
import org.autonlab.activesearch.ActiveSearchConstants;

import java.util.ArrayList;
import java.util.LinkedList;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class GenerateCompabilityMatrix {

	static EmailSimilarity[] emailData;
	static int threadCount;

	public static void main(String[] args) {
		/*
	DataConnectionMySQL dataConnection = new DataConnectionMySQL();
	int emailCount = dataConnection.getTotalEmailCount();
	dataConnection.closeConnection();

	DoubleMatrix xmappedMatrix = readFile("/home/tw/8000_enron/Xmapped.txt", emailCount, 0);
	DoubleMatrix similaritySumMatrix = readFile("/home/tw/8000_enron/A.txt", emailCount, 1);
	DoubleMatrix labelsMatrix = readFile("/home/tw/8000_enron/labels.txt", emailCount, 1);

	SearchMain aSearch = new SearchMain(xmappedMatrix, similaritySumMatrix, 
					    ActiveSearchConstants.SEARCH_MAIN_DIMENSIONS, 
					    ActiveSearchConstants.SEARCH_MAIN_ALPHA,
					    ActiveSearchConstants.SEARCH_MAIN_OMEGA,
					    -1,
					    ActiveSearchConstants.SEARCH_MAIN_OFFSET_FLAG,
					    labelsMatrix,
					    ActiveSearchConstants.MODE_START_RANDOM);
	aSearch.calculate(100);
		 */

		// This code is for generating the similarity matrix in a sparse matrix format 

		int i;

		if (args[0] == null || Integer.parseInt(args[0]) < 1) {
			System.out.println("Error: need a positive number of threads to start on command line");
			return;
		}
		threadCount = Integer.parseInt(args[0]);
		System.out.println("num threads is " + threadCount);

		emailData = new EmailSimilarity[threadCount];
		for (i = 0; i < threadCount; i++) {
			emailData[i] = new EmailSimilarity(i, 1);
		}
		emailData[0].dontDoWrites();
		emailData[0].initSimilarityData(threadCount);
		for (i = 0; i<threadCount; i++) {
			emailData[i].buildMatrix();
		}
		System.out.println("Waiting for threads to complete");
		emailData[0].waitThreads();
		System.out.println("Threads done");

		DoubleMatrix similaritySumMatrix = emailData[0].getMatrixFile();
		for (i = 0; i < threadCount; i++) {
			emailData[i] = null;
		}
		emailData = null;
		EmailSimilarity.clearStaticValues();

		GlapEigenmap foo = new GlapEigenmap(ActiveSearchConstants.SEARCH_MAIN_DIMENSIONS, similaritySumMatrix);

		/*
	DataConnectionMySQL dataConnection = new DataConnectionMySQL();
	int emailCount = dataConnection.getTotalEmailCount();
	dataConnection.closeConnection();


	emailData = new EmailSimilarity[1];
	emailData[0] = new EmailSimilarity(0, 0);
	emailData[0].readMatrixFile(emailCount, "/home/tw/h/ActiveSearch/CytoscapeApp/src/similarity_13914_combo.out");
	GlapEigenmap foo = new GlapEigenmap(2000, emailData[0].getMatrixFile());
		 */
	}

	public static DoubleMatrix readFile(String filename, int rows, int isVector) {
		BufferedReader br = null;
		DoubleMatrix tempMatrix = null;
		int count = 0;

		if (isVector > 0) {
			tempMatrix = new DoubleMatrix(rows);
			System.out.println("File has 1 column");
		}

		try {
			int i;
			String sCurrentLine;
			String[] sParts;
			br = new BufferedReader(new FileReader(filename));
			System.out.print("Importing matrix ");
			while ((sCurrentLine = br.readLine()) != null) {

				if (count % 2000 == 0) {
					System.out.print(".");
				}

				if (isVector == 0) {
					sParts = sCurrentLine.split(" ");
					if (tempMatrix == null) {
						System.out.println("File has " + sParts.length + " columns");
						tempMatrix = new DoubleMatrix(rows, sParts.length);
					}
					for (i = 0; i < sParts.length; i++) {
						tempMatrix.put(count, i, Double.parseDouble(sParts[i]));
					}
				}
				else {
					tempMatrix.put(count, Double.parseDouble(sCurrentLine));
				}
				count++;
			}
			System.out.println("");
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to read file " + filename);
		} finally {
			try {
				if (br != null)br.close();
			} catch (IOException ex) {
				ex.printStackTrace();
				throw new RuntimeException("Failed to read file " + filename);
			}
		}
		return tempMatrix;
	}
}
