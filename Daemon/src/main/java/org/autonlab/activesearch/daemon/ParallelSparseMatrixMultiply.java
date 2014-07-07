package org.autonlab.activesearch;

import org.apache.commons.math3.linear.*;

import org.jblas.*;

public class ParallelSparseMatrixMultiply implements Runnable {
    int _threadID = -1;
    static int _threadCount = 0;
    static OpenMapRealMatrix[] _matrixAArray = null;
    static OpenMapRealMatrix _matrixB = null;
    static OpenMapRealMatrix[] _matrixResult = null;
    static Thread[] _threadArray = null;
    static DoubleMatrix _matrixRet = null;
    static int _rows = 0;

    /**
     * This multiplies matrixA by its transpose. I tried to write this
     * code so that another constructor could be written to multiply 
     * any two matrices but I don't need that functionality right now
     */
    public ParallelSparseMatrixMultiply(int threadID, int threadCount, DoubleMatrix matrixIn)
    {
	int i;
	int j;
	_threadID = threadID;

	if (_threadCount == 0) {
	    _threadCount = threadCount;

	    _matrixAArray = new OpenMapRealMatrix[threadCount];
	    _matrixB = new OpenMapRealMatrix(matrixIn.columns, matrixIn.rows);
	    _matrixResult = new OpenMapRealMatrix[threadCount];
	    _threadArray = new Thread[threadCount];

	    /* ignoring this unlikely case saves us some testing */
	    if (matrixIn.rows < threadCount) {
		throw new RuntimeException("Number of threads is greater than number of rows in the matrix. This is not supported!");
	    }

	    int interval = matrixIn.rows / threadCount;
	    if ((matrixIn.rows % threadCount) != 0) { 	// round up so the last thread doesn't drop rows
		interval++;
	    }

	    for (i = 0; i < threadCount; i++) {
		int rowCount = (Math.min(interval * (i+1), matrixIn.rows)) - (interval * i) ;
		_matrixAArray[i] = new OpenMapRealMatrix(rowCount, matrixIn.columns);
	    }

	    int oldWorkingThreadID = -1;
	    int iArray = 0;
	    for (i = 0; i < matrixIn.rows; i++) {
		int workingThreadID = i / interval;
		if (oldWorkingThreadID != workingThreadID) {
		    oldWorkingThreadID = workingThreadID;
		    iArray = 0;
		}
		for (j=0;j<matrixIn.columns;j++) {
		    // If an email has no words, there will be NaN values due to the division by zero. Reset those NaN values to zero proper
		    if (Double.isNaN(matrixIn.get(i,j))) {
			System.out.println("calculation for email " + i + "," + j + " resulted in NaN. Resetting to zero (this is ok)");
			matrixIn.put(i, j, 0.0);
		    }
		    if (matrixIn.get(i,j) != 0.0) {
			_matrixAArray[workingThreadID].setEntry(iArray,j, matrixIn.get(i,j));
			_matrixB.setEntry(j,i, matrixIn.get(i,j));
		    }
		}
		iArray++;
	    }
	    // try to limit the max amount of memory we use by clearing matrixIn before we create _matrixRet
	    _rows = matrixIn.rows;
	    matrixIn = null;
	    _matrixRet = new DoubleMatrix(_rows, _rows);
	}
    }

    public ParallelSparseMatrixMultiply(int threadID)
    {
	_threadID = threadID;
    }

    /* to free up memory when this processing step is done */
    public static void clearStaticValues()
    {
	_threadCount = 0;
	_matrixAArray = null;
	_matrixB = null;
	_matrixResult = null;
	_threadArray = null;
	_matrixRet = null;
	_rows = 0;
    }

    public void multiply() 
    {
	_threadArray[_threadID] = new Thread(this);
	_threadArray[_threadID].start();
    }

    public void run() {
	int i;
	int j;

	int interval = _rows / _threadCount;
	if ((_rows % _threadCount) != 0) { 	// round up so the last thread doesn't drop rows
	    interval++;
	}
	int startID = interval * _threadID;
	int endID = Math.min(interval * (_threadID+1), _rows);

	System.out.println("Thread " + _threadID + " started from " + startID + " to (but not including)" + endID);

	_matrixResult[_threadID] = _matrixAArray[_threadID].multiply(_matrixB);

	System.out.println("Thread " + _threadID + " done with multiply. Now doing copy to return matrix");
	// copy the subset's result into the return matrix
	int iArray = 0;
	for (i = startID; i < endID; i++) {
	    for (j = 0; j < _matrixResult[_threadID].getColumnDimension(); j++) {
		_matrixRet.put(i, j, _matrixResult[_threadID].getEntry(iArray, j));
	    }
	    iArray++;
	}
	System.out.println("Thread " + _threadID + " done.");
    }

    public static DoubleMatrix getResult() {
	int i;
	for (i = 0; i < _threadCount; i++) {
	    try {
		_threadArray[i].join();
	    } catch (InterruptedException e) {
		e.printStackTrace();
	    }
	}
	return _matrixRet;
    }
}