package org.autonlab.activesearch;

import org.autonlab.activesearch.DataConnectionMySQL;
import org.jblas.*;

import java.io.*;

import java.util.*;


//Eigen.symmetricEigenvectors(DoubleMatrix A) 
// an array of DoubleMatrix objects containing the eigenvectors stored as the columns of the first matrix, and the eigenvalues as diagonal elements of the second matrix.
// first part is getV, second part is getD

public class GlapEigenmap {
    int dimensions;
    int emailCount;
    DoubleMatrix eigenMap;

    /**
     * @param eigenmapFile Read in the Eigenmap from a file rather than calculating it
     */
    public GlapEigenmap(String eigenmapFile) {
	int i;
	BufferedReader br = null;

	DataConnectionMySQL dataConnection = new DataConnectionMySQL();
	emailCount = dataConnection.getTotalEmailCount();
	dataConnection.closeConnection();

	try {
	    String sCurrentLine;
	    String[] sParts;
	    br = new BufferedReader(new FileReader(eigenmapFile));
	    System.out.print("Importing Eigenmap");
	    int rowCount = 0;
	    while ((sCurrentLine = br.readLine()) != null) {
		if (rowCount % (emailCount/500) == 0) {
		    System.out.print(".");
		}
		sParts = sCurrentLine.split(" ");

		if (rowCount == 0) {
		    dimensions = sParts.length;
		    eigenMap = new DoubleMatrix(emailCount, dimensions);
		}

		for (i = 0; i < dimensions; i++) {
		    eigenMap.put(rowCount, i, Integer.parseInt(sParts[i]));
		}
		rowCount++;
	    }
	    System.out.println("\nDone importing Eigenmap");
	} catch (IOException e) {
	    e.printStackTrace();
	    throw new RuntimeException("Failed to import Eigenmap from file");
	} finally {
	    try {
		if (br != null)br.close();
	    } catch (IOException ex) {
		ex.printStackTrace();
		throw new RuntimeException("Failed to close Eigenmap file");
	    }
	}
    }

    /**
     * @param myDim Number of leading dimensions for the Eigenmap
     * @param similarityMatrix Similarity matrix to use for Eigenmap
     */
    public GlapEigenmap(int myDim, DoubleMatrix similarityMatrix) {
	DataConnectionMySQL dataConnection = new DataConnectionMySQL();
	emailCount = dataConnection.getTotalEmailCount();
	dataConnection.closeConnection();
	//	myDim = 8;
	dimensions = myDim;

	DoubleMatrix myLMatrix;
	DoubleMatrix myXMatrix;
	DoubleMatrix myXMatrixSorted;
	DoubleMatrix[] tempMatrix;
	DoubleMatrix myLambda;
	DoubleMatrix myLambdaTemp;
	DoubleMatrix myLambdaSorted;
	DoubleMatrix myW;

	int i;
	int j;

	//	emailCount = 10;
	//similarityMatrix = similarityMatrix.getRange(0,10,0,10);

	// L = diag(sum(A,2)) - A;
	myLMatrix = (DoubleMatrix.diag(similarityMatrix.rowSums())).subi(similarityMatrix);

	if (dimensions > similarityMatrix.columns) {
	    throw new RuntimeException("Error: required dimension larger than size of similarity matrix!");
	}

	// [X lambda] = eig(full(L));
	System.out.println("Calculate eigen decomposition on " + myLMatrix.rows + "x" + myLMatrix.columns);
	tempMatrix = Eigen.symmetricEigenvectors(myLMatrix);
	myXMatrix = tempMatrix[0];
	myLambda = tempMatrix[1];
	System.out.println("Resulting X " + myXMatrix.rows + "x" + myXMatrix.columns);
	System.out.println("Resulting L " + myLambda.rows + "x" + myLambda.columns);
	DoubleMatrix foo = myXMatrix.rowSums().columnSums();
	System.out.println("X:" + foo.get(0));
	/*
	for (i = 0; i < myXMatrix.rows; i++) {
	    for(j=0;j<myXMatrix.columns;j++){
		System.out.print(myXMatrix.get(i,j) + "  ");
	    }
	    System.out.println("");
	    }*/

	// lambda = diag(lambda);
	myLambda = myLambda.diag();

	DoubleMatrix foo2 = myLambda.rowSums().columnSums();
	System.out.println("lambda:" + foo2.get(0));

	/*
	 * [lambda, perm] = sort(lambda, 'ascend');
	 * Create a basic ArrayList where ArrayList[index] -> index. Then we'll sort 
	 * this ArrayList based on the values in myLambda[index]. This will give us
	 * ArrayList[new sorted index] -> old index, a permutation that we can use to 
	 */
	int[] myLambdaTempIndexes = myLambda.sortingPermutation();
	myLambdaSorted = myLambda.sort();

	/*
	 * th_zero = 1/size(A,1)/1e3
	 * b = sum(lambda < th_zero)
	 */
	double th_zero = 1/similarityMatrix.columns/1000;
	int b = 0;
	for (i = 0; i < myLambdaSorted.length; i++) {
	    if (myLambdaSorted.get(i) < th_zero) {
		b++;
	    }
	}

	/*
	 * w = 1./sqrt(lambda(2:(d+1)));
	 *
	 * w = 1./sqrt(lambda((b+1):d))
	 * Note: Java array indexes start at 0 so we'll go from myLambda[b -> d-1] not b+1:d
	 * getRange returns [a,b) so to get 1->d we pass 1->d+1
	 */
	myXMatrixSorted =  new DoubleMatrix(emailCount, emailCount);
	myW = new DoubleMatrix(dimensions);
	int currentIndex = 0;

	for (i = b; i < dimensions; i++) {
	    Double temp = Math.pow(Math.sqrt(myLambdaSorted.get(i)), -1);
	    myW.put(currentIndex, temp);
	    currentIndex++;
	}

	for (i = 0; i < dimensions; i++) {
	    myXMatrixSorted.putColumn(i, myXMatrix.getColumn(myLambdaTempIndexes[i]));
	}

	myXMatrixSorted = myXMatrixSorted.getRange(0, myXMatrixSorted.rows, 1, dimensions+1);

	myXMatrix = myXMatrixSorted;
	myXMatrixSorted = null;

	System.out.println("X is now " + myXMatrix.rows + "x" + myXMatrix.columns);
	System.out.println("W is " + myW.length);

	// X = [X(:perm(1:b)) bsxfun(@times, X(:,perm(b+1:d)), reshape(w,1,length(w)))];
	for (i = 0; i < myXMatrix.rows; i++) {
	    for (j = b; j < dimensions; j++) {
		myXMatrix.put(i, j, myXMatrix.get(i, j) * myW.get(j));
	    }
	}

	GlapEigenmap.write(myXMatrix, ActiveSearchConstants.X_MATRIX, emailCount, dimensions);
	DoubleMatrix bMatrix = new DoubleMatrix(1,1);
	bMatrix.put(0,0,b);
	GlapEigenmap.write(bMatrix, ActiveSearchConstants.b_MATRIX, 1, 1);
	//GlapEigenmap.write(myW, "w_matrix_tw.out", dimensions, 1);

    }

    public DoubleMatrix getEigenmap() {
	return eigenMap;
    }

    public static void write(DoubleMatrix matrix, String filename, int rows, int columns) {
	int i;
	int j;

	BufferedWriter bw = null;
	File file = new File(filename);
	try {
	    if (!file.exists()) {
		file.createNewFile();
	    }
	    FileWriter fw = new FileWriter(file.getAbsoluteFile());
	    bw = new BufferedWriter(fw);

	    for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
		    bw.write(matrix.get(i, j) + " ");
		}
		bw.newLine();
	    }
	    bw.close();
	} catch (IOException e) {
	    e.printStackTrace();
	    throw new RuntimeException("Error writing sparse matrix to disk");
	}
    }
}

    