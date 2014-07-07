package org.autonlab.activesearch;

import org.autonlab.activesearch.SparseMatrix;
import java.util.*;
import org.jblas.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
public class SparseMatrix {
    int size = 0;
    ArrayList<TreeMap<Integer, Double>> matrix;

    public SparseMatrix(int size) {
	int i;

	this.size = size;
	matrix = new ArrayList<TreeMap<Integer, Double>>(size);
	for (i = 0; i < size; i++) {
	    matrix.add(new TreeMap<Integer, Double>());
	}
    }

    /**
     * Insert a value into the sparse matrix
     *
     * @in row the row associated with the value
     * @in column the column associated with the value
     * @in value the value to store
     */
    public void put(int row, int column, double value) {
	if (row < 0 || row >= size) {
	    throw new RuntimeException("Illegal row");
	}
	if (column < 0 || column >= size) {
	    throw new RuntimeException("Illegal column");
	}

	// If the value is zero we don't store it, hence the sparseness
	if (value > 0.0) {
	    matrix.get(row).put(new Integer(column), new Double(value));
	}
    }

    /**
     * Retrieve a value from the sparse matrix. If the value is not 
     * found, we return 0
     *
     * @in row the row to retrieve
     * @in column the column to retrieve
     *
     * @return the value from the sparse matrix
     */
    public double get(int row, int column) {
	Double val;

	if (row < 0 || row >= size) {
	    throw new RuntimeException("Illegal row");
	}
	if (column < 0 || column >= size) {
	    throw new RuntimeException("Illegal column");
	}

	val = matrix.get(row).get(new Integer(column));
	if (val == null) {
	    return 0.0;
	}
	else {
	    return val.doubleValue();
	}
    }

    /**
     * Calculate the rowsum and return it in DoubleMatrix form
     *
     */
    public DoubleMatrix rowSums() {
	DoubleMatrix retMatrix = new DoubleMatrix(size);
	int i;

	for (i = 0; i < size; i++) {
	    double rowSum = 0.0;
	    for (Map.Entry<Integer, Double> entry: matrix.get(i).descendingMap().entrySet()) {
		rowSum += entry.getValue().doubleValue();
	    }
	    retMatrix.put(i, rowSum);
	}

	return retMatrix;
    }
	
    /**
     * Write the sparse matrix to a file in the form i+1, j+1, value
     * We add the +1 to each index because Matlab indices start at 1 not zero
     * Assumes that the matrix is symmetrical and only dumps the upper half
     */
    public void write(String filename) {
	int i;
	BufferedWriter bw = null;
	File file = new File(filename);
	try {
	    if (!file.exists()) {
		file.createNewFile();
	    }
	    FileWriter fw = new FileWriter(file.getAbsoluteFile());
	    bw = new BufferedWriter(fw);

	    for (i = 0; i < size; i++) {
		/* 
		 * The matrix is symmetrical so we only need to dump half of it. We'll dump
		 * the upper half by using all values of i and only values of j >= i.
		 */
		for (Map.Entry<Integer, Double> entry: matrix.get(i).descendingMap().entrySet()) {
		    int j = entry.getKey();
		    if (j >= i) {
			double thisValue = entry.getValue().doubleValue();
			bw.write((i+1) + " " + (j+1) + " " + thisValue);
			bw.newLine();
		    }
		    else {
			break;
		    }
		}
	    }
	    bw.close();
	} catch (IOException e) {
	    e.printStackTrace();
	    throw new RuntimeException("Error writing sparse matrix to disk");
	}
    }

    /**
     * Write a DoubleMatrix to a file in sparse matrix format so Matlab can read it
     * We add +1 to each index because Matlab indices start at 1 not zero
     * Assumes that the matrix is symmetrical and only dumps the upper half
     */
    public static void writeDoubleMatrix(DoubleMatrix matrix, String filename) {
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

	    for (i = 0; i < matrix.rows; i++) {
		for (j = i; j < matrix.columns; j++) {
		    if (matrix.get(i,j) != 0.0) {
			bw.write((i+1) + " " + (j+1) + " " + matrix.get(i,j));
			bw.newLine();
		    }
		}
	    }
	    bw.close();
	} catch (IOException e) {
	    e.printStackTrace();
	    throw new RuntimeException("Error writing sparse matrix to disk");
	}
    }
}
