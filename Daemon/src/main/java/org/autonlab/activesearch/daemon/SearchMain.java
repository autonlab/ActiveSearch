package org.autonlab.activesearch;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.Solve;

import org.jblas.ranges.RangeUtils;
import org.jblas.ranges.Range;

import java.util.ArrayList;
import java.util.Collections;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This class is ported from a Matlab file that computed Active Search in a loop with no user interaction
 * (see the compute() function). Sometimes the Java code looks awkward because the two languages are so
 * different. Adding the user interactivity exacerbates the awkwardness
 */
public class SearchMain {

    // Making these static saves us a little bit of processing time switching between modes
    static DoubleMatrix eigenMapX = null;
    static DoubleMatrix eigenMapXp = null;
    static DoubleMatrix eigenMapXpTranspose = null;
    static DoubleMatrix similarityDegree = null;

    // There are a few bit arrays stored as DoubleMatrix to simplify the code even if it wastes a little space
    DoubleMatrix labels;
    static int dimensions;

    double alpha;
    int emailCount;

    DoubleMatrix in_train;

    int best_ind;

    static int offset_flag = -1;
    double omega0;
    double pai = 0.05;
    double eta = 0.5;

    DoubleMatrix yp;
    double lamb;
    double rVal;
    double cVal;
    static DoubleMatrix sqd = null;

    /* for now we assume this will be 1 for the forseeable future */
    int num_initial = 1; //number of initial target points

    static int start_point = -1;
    int[] hits;
    int[] selected;

    DoubleMatrix C;
    DoubleMatrix h;
    DoubleMatrix f;

    int mode;

    
    /**
     * @param eigenMap The precomputed eigen decomposition of the similarity matrix
     * @param similarityDeg
     * @param dim Number of dimensions to use
     * @param myAlpha This value is divided by 1000 befure it is used
     * @param omega omega0 value. A negative number will result in a default value (1/emailCount)
     * @param startp Which email to start from. Negative number = random. Set a value for a repeatable test case or to compare to Matlab's output
     * @param myOffsetFlag 1 or 0, whether or not to use an offset
     * @param myLabels The labels vector, stored as a DoubleMatrix to simplify the code
     * @param myMode The internal processing mode. See ActiveSearchConstants.java
     */
    public SearchMain(DoubleMatrix eigenMap, DoubleMatrix similarityDeg, int dim, int myAlpha, double omega, int startp, int myOffsetFlag, DoubleMatrix myLabels, int myMode, int nConnComp) { 
	/*
	 * Here are some notes that will be useful for understanding the code:
	 *
	 * The Matlab code we're porting from has a starting index of 1 whereas in Java the starting index is 0, so you'll see things like emailCount-1 and dimensions-1.
	 *     Sometimes the Matlab code calls for dimensions-1 then you'll see dimensions-2 and so on. See the note below for getRange where this differs!
	 *
	 * The jblas library has two groups of operators. Things like A.mmul(B) will matrix multiple A*B and return a new matrix C. We use this type for
	 *     calculations involving our precomputed eigenMap that we'll use lots of times. There's also many of these functions also have a version 
	 *     with a trailing "i" (like mmuli) which does the same calculation but overwrites the original matrix. That is, A*B would overwrite A with the
	 *     result. This saves memory in that we don't create a third matrix. Both functions typically return the calculated matrix for convenience.
	 *
	 * Note the difference between A.mmul(B) and A.mul(B) where the former is matrix multiplcation and the latter is element-wise multiplication
	 *
	 * getRange returns a copy of the matrix's range  (to me the documentation was unclear if it was a copy or a subset of the original)
	 *     Also, getRange(ra, rb, ca, cb) returns the matrix from [ra, rb) to [ca, cb). The documentation lies about [ra, rb][ca, cb].
         *     So, even though most java calls will be to matlabIndex-1, getRange will be getRange(matlabIndexA-1, matlabIndexB, matlabIndexC-1, matlabIndexD)
	 *
	 * DoubleMatrix.add claims to be in-place but it is not (it returns a copy with the sum and leaves the original alone)
	 */
	int i;
	int j;

	if (similarityDegree == null) {
	    similarityDegree = similarityDeg;
	}

	alpha = ((double)myAlpha) / 1000.0;
	omega0 = omega;
	if (mode != ActiveSearchConstants.MODE_SHOW_LAST_SEED) {
	    start_point = startp;
	}

	labels = myLabels;
	mode = myMode;

	if ((mode == ActiveSearchConstants.MODE_SHOW_EDGE_EMAILS || mode == ActiveSearchConstants.MODE_SHOW_LAST_SEED || mode == ActiveSearchConstants.MODE_DO_NOTHING_USER_WILL_PICK_SEED) 
	    && start_point != -1) {
	    System.out.println("setting label for seed " + start_point);
	    labels.put(start_point,1);
	}

	if (dim < 1) {
	    throw new RuntimeException("Illegal dimension " + dim);
	}
	if (alpha < 0) {
	    throw new RuntimeException("Illegal alpha " + alpha);
	}

	if (eigenMap.rows != similarityDegree.length) {
	    throw new RuntimeException("Eigenmap number of rows " + eigenMap.rows + " didn't match similarity degree length " + similarityDegree.length);
	}


	if (dim > eigenMap.columns) {
	    throw new RuntimeException("Dimension argument " + dim + " is greater than number of dim in Eignemap " + eigenMap.columns);
	}

	emailCount = eigenMap.getRows();

	if (omega0 < 0) {
	    omega0 = 1.0/((double)emailCount);
	}

	if (eigenMapX == null || ((dim + (myOffsetFlag==0 ? 0 : 1)) != dimensions)) {
	    eigenMapX = eigenMap;
	    dimensions = dim;

	    // Drop the extra dimensions from the Eigenmap
	    eigenMapX = eigenMapX.getRange(0, emailCount, 0, dimensions);

	    // Append 1 to feature vector if linear regression includes an offset
	    if (myOffsetFlag > 0) {
		dimensions++;
		DoubleMatrix tempX = new DoubleMatrix(emailCount, dimensions);
		for (i = 0; i < eigenMapX.rows; i++) {
		    for (j = 0; j < eigenMapX.columns; j++) {
			tempX.put(i, j, eigenMapX.get(i,j));
		    }
		}
		for (i = 0; i < emailCount; i++) {
		    tempX.put(i, dimensions-1, 1.0);
	    }
		eigenMapX = tempX;
	    }
	}

	if (sqd == null) {
	    sqd = MatrixFunctions.sqrt(similarityDeg);
	}

	lamb = (1-eta)/eta;
	rVal = lamb * omega0;
	cVal = 1/(1-rVal);

	if (eigenMapXp == null || offset_flag != myOffsetFlag) {
	    eigenMapXp = eigenMapX.dup();
	    System.out.println(emailCount + " and " + dimensions);
	    for (i = 0; i < emailCount; i++) {
		for (j = 0; j < dimensions; j++) {
		    eigenMapXp.put(i, j, eigenMapXp.get(i,j) * sqd.get(i));
		}
	    }
	}
	offset_flag = myOffsetFlag;

	if (eigenMapX.rows != labels.length) {
	    throw new RuntimeException("Eigenmap number of rows " + eigenMapX.rows + " didn't match labels array length " + labels.length);
	}

	in_train = new DoubleMatrix(emailCount);

	if (start_point < 0) {
	    start_point = pickRandomLabeledEmail();
	}

	//  the user is interested in the start_point so we don't need to do the other calculations
	if (mode == ActiveSearchConstants.MODE_SHOW_SELECT_SEED) {
	    return;
	}

	in_train.put(start_point,1);
	best_ind = start_point;
	System.out.println("Start point " + best_ind);

	yp = new DoubleMatrix(emailCount);
	for (i = 0; i < emailCount; i++) {
	    yp.put(i, labels.get(i) * sqd.get(i));
	}

	if (eigenMapXpTranspose == null) {
	    eigenMapXpTranspose = eigenMapXp.transpose();
	}

	// C = r*(Xp'*Xp) + (1-r)*(Xp(best_ind,:)'*Xp(best_ind,:)) + lamb*diag([zeros(b,1); ones(d-b,1)]);
	//       ^temp1 ^   ^------------- temp2 ----------------^   ^------------- temp3 ---------------^
	DoubleMatrix temp1 = eigenMapXpTranspose.mul(rVal).mmul(eigenMapXp);
	DoubleMatrix temp2 = eigenMapXp.getRow(best_ind).transpose().mmul(eigenMapXp.getRow(best_ind)).mul(1-rVal);
	DoubleMatrix temp3 = DoubleMatrix.eye(dimensions);
	for (i = 0; i < nConnComp; i++) {
	    temp3.put(i, i, 0.0);
	}
	temp3.muli(lamb);

	C = temp1.addi(temp2).addi(temp3);
	// calculate the inverse
	C = Solve.solve(C, DoubleMatrix.eye(dimensions));

	// h = sum((Xp*C).*Xp,2);
	h = eigenMapXp.mmul(C).mul(eigenMapXp).rowSums();

	// f = X * (C * (r*Xp'*sqd*pai + Xp(best_ind,:)'*(yp(best_ind)-r*sqd(best_ind)*pai)));
	//               ^-- temp1 --^   ^---------------- temp2 ---------------------------^
	temp1 = eigenMapXpTranspose.mmul(sqd).mul(pai*rVal);
	temp2 = eigenMapXp.getRow(best_ind).transpose().mul(yp.get(best_ind)-(sqd.get(best_ind)*rVal*pai));
	f = eigenMapX.mmul(C.mmul(temp1.add(temp2)));
    }


    /**
     * Randomly pick 1 target point as the first point
     * We do it this way because in the future we may want to generate a sequence of indices rather than just one
     *
     * @return an email index representing an email with a positive label
     */
    public int pickRandomLabeledEmail() {
	int i;

	ArrayList<Integer> temp = new ArrayList<Integer>(emailCount);
	for (i = 0; i < emailCount; i++) {
	    if (labels.get(i) != 0) {
		temp.add(new Integer(i)); // store the index i rather than than the value of labels.get(i)
	    }
	}
	temp.trimToSize();
	Collections.shuffle(temp);
	if (num_initial != 1) {
	    throw new RuntimeException("Error: this code was written assuming that num_initial was 1. Please update the code");
	}
	return temp.get(0).intValue();
    }

    /**
     * Perform a series of Active Search calculations. This loop mimics the original Matlab code. It is not used
     * in interactive mode
     *
     * @param numEvaluations
     * @return the highest scoring email
     */
    public int calculate (int numEvaluations) {
	int i;
	BufferedWriter bw = null;
	File file = null;

	hits = new int[numEvaluations + 1];
	selected = new int[numEvaluations + 1];

	/* 1 is the lowest index in Matlab but we use zero in Java */
	hits[0] = 1;
	selected[0] = best_ind;

	if (numEvaluations > 1) {
	    int random_seed = 0;
	    String filename = (Long.toString(System.currentTimeMillis() / 1000l)) + "_seed" + Integer.toString(random_seed) + "_alpha" + Double.toString(alpha) + "_offset" + Integer.toString(offset_flag) + "_omega0-" + Double.toString(omega0) + "_d" + dimensions + ".txt";
	    file = new File(filename);
	    try {
		if (!file.exists()) {
		    file.createNewFile();
		}
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		bw = new BufferedWriter(fw);
	    } catch (IOException e) {
		e.printStackTrace();
		throw new RuntimeException("Error opening Active Search file");
	    }
	}
	else {
	    return -1;
	}

	int main_loop_i;
	for (main_loop_i = 0; main_loop_i < numEvaluations; main_loop_i++) {
	    System.out.println("Main loop " + main_loop_i + " / " + numEvaluations);
	    long tic = System.nanoTime();

	    int best_ind = getNextEmail();

	    if (countOnes(labels.get(in_train)) == countOnes(labels)) {
		System.out.println("evals: " + main_loop_i + " trained: " + countOnes(labels.get(in_train)));
		break;
	    }

	    updateParams(best_ind);

	    long toc = System.nanoTime();

	    selected[main_loop_i+1] = best_ind;
	    hits[main_loop_i+1] = countOnes(labels.get(in_train));

	    if (numEvaluations > 1) {
		try {
		    bw.write((main_loop_i+1) + " " + hits[main_loop_i+1] + " " + ((float)((toc - tic)/1e9)) + " " + (selected[main_loop_i+1]+1));
		    bw.newLine();
		} 
		catch (IOException e) {
		    e.printStackTrace();
		    throw new RuntimeException("Error writing sparse matrix to disk");
		}
	    }
	}

	System.out.println("Selected");
	for (i = 0; i < numEvaluations+1; i++) {
	    System.out.print(selected[i] + " ");
	}
	System.out.println("\nHits");
	for (i = 0; i < numEvaluations+1; i++) {
	    System.out.print(hits[i] + " ");
	}
	System.out.println();

	if (numEvaluations > 1) {
	    try {
		bw.close();
	    } catch (IOException e) {
		e.printStackTrace();
		throw new RuntimeException("Error writing sparse matrix to disk");
	    }
	}

	return selected[1];
    }

    /**
     * Count the number of ones in a vector
     *
     * @param inMatrix the vector to check
     * @return the number of ones
     */
    public int countOnes(DoubleMatrix inMatrix) {
	int i;
	int count = 0;
	for (i = 0; i < inMatrix.length; i++) {
	    if (inMatrix.get(i) == 1.0) {
		count++;
	    }
	}
	return count;
    }

    /**
     * Do the computation needed to suggest an email to the user
     *
     * @return email ID suggestion
     */
    public int getNextEmail() {
	int i;

	if (eigenMapXpTranspose == null) {
	    eigenMapXpTranspose = eigenMapXp.transpose();
	}

	DoubleMatrix best_ind_vector; // differentiate from best_ind, which the matlab code uses in both places

	// test_ind = ~in_train;
	DoubleMatrix test_ind = in_train.not();

	// change = ((((test_ind'*X)*C)*Xp')' - (h./sqd)) .* sqd .* ((1-r*pai)*c-f) ./ (c+h);
	//          ^ ---------------- temp1 -----------^
	DoubleMatrix temp1 = test_ind.transpose().mmul(eigenMapX).mmul(C).mmul(eigenMapXpTranspose).transpose().sub(h.div(sqd));
	DoubleMatrix temp2 = sqd.mul(f.rsub((1-rVal*pai)*cVal)).div(h.add(cVal));
	DoubleMatrix change = temp1.mul(temp2);

	// f_bnd = min(max(f(test_ind),0),1);
	// get(test_ind) returns all values f[i] where test_ind[i] is nonzero 
	DoubleMatrix f_bnd = f.get(test_ind).max(DoubleMatrix.zeros(test_ind.rows, test_ind.columns)).min(DoubleMatrix.ones(test_ind.rows, test_ind.columns));

	// score = f_bnd + alpha*f_bnd.*max(change(test_ind),0);
	/* 
	 * Here we only consider rows where test_ind=1.  As a result
	 * the score vector is smaller than the total number of emails
	 * by the number of labeled emails
	 */
	DoubleMatrix tempScore = change.get(test_ind);
	for (i = 0; i < tempScore.length; i++) {
	    if (tempScore.get(i) < 0.0) {
		tempScore.put(i, 0.0);
	    }
	}
	DoubleMatrix score = f_bnd.add(f_bnd.mul(tempScore).mul(alpha));

	// [best_score best_ind] = max(score);
	int best_ind = score.argmax();

	/*
	 * best_ind is an index into the score vector, which is
	 * smaller than the number of emails so we need to figure out
	 * what it maps to in the original email list. RangeUtils.find
	 * does this for us but it results in a Range instead of Doublematrix
	 * so we have to do a conversion.
	 */
	Range temp_ind_list = RangeUtils.find(test_ind);
	DoubleMatrix temp_ind_list_matrix = new DoubleMatrix(temp_ind_list.length());
	for (i = 0; i < temp_ind_list.length(); i++) {
	    temp_ind_list_matrix.put(i, temp_ind_list.value());
	    temp_ind_list.next();
	}

	//System.out.println("best is " + best_ind + " : " + score.get(best_ind));

	/* make best_ind map to the original set of emails rather than the reduced set of score emails */
	best_ind = (int)temp_ind_list_matrix.get(best_ind);

	return best_ind;
    }

    /**
     * After a user labels a new email, update some matrices
     *
     * @param best_ind the email whose label was set
     */
    private void updateParams(int best_ind) {	
	// CXp = C * Xp(best_ind,:)';
	DoubleMatrix CXp = C.mmul(eigenMapXp.getRow(best_ind).transpose());

	//f = f + X * ( (CXp*((yp(best_ind)-r*sqd(best_ind)*pai)*c - sqd(best_ind)*f(best_ind)) / (c+h(best_ind))) );
	//               ^----------------------------------- temp1 ---------------------------------------------^   
	DoubleMatrix temp1 = CXp.mul((((yp.get(best_ind)-(rVal*sqd.get(best_ind)*pai)) * cVal) - (sqd.get(best_ind)*f.get(best_ind)))).div(cVal+h.get(best_ind));
	f.addi(eigenMapX.mmul(temp1));

	//C = C - (CXp*CXp')/(c+h(best_ind));
	C.subi(CXp.mmul(CXp.transpose()).div(cVal+h.get(best_ind)));

	//h = h - (Xp*CXp).^2 / (c+h(best_ind));
	h.subi(MatrixFunctions.pow(eigenMapXp.mmul(CXp),2).div(cVal+h.get(best_ind)));
    }

    /**
     * After a user relabels a previously labeled email, update some matrices
     *
     * @param best_ind the email whose label was set
     * @param oldVal The label value before the user changed it again
     * @param newVal The label value after the user changed it
     */
    private void updateParamsReset(int best_ind, double oldVal, double newVal) {
	// CXp = C * Xp(best_ind,:)';
	DoubleMatrix CXp = C.mmul(eigenMapXp.getRow(best_ind).transpose());

	f.addi(eigenMapX.mmul(CXp).mul(newVal - oldVal));

	// only update f in this case and not C or h
    }

    /**
     * Set a new label
     *
     * @param index The email ID that is set
     * @param value The user's impression of the email. 1=interesting 0=boring
     */
    public void setLabel(int index, double value) {
	in_train.put(index, 1);

	labels.put(index, value);
	yp.put(index, labels.get(index) * sqd.get(index));

	updateParams(index);
    }

    /**
     * Set a previously set label
     *
     * @param index The email ID that is set
     * @param value The user's impression of the email. 1=interesting 0=boring
     * @return the old label value before it was set again
     */
    public double resetLabel(int index, double value) {
	double oldVal = labels.get(index);

	labels.put(index, value);
	yp.put(index, labels.get(index) * sqd.get(index));

	updateParamsReset(index, oldVal, value);

	return oldVal;
    }

    public double getLabel(int email) {
	return labels.get(email);
    }

    /**
     * return the email ID that seeded this search sequence
     *
     * @return email ID
     */
    public static int getStartPoint() {
	return start_point;
    }

    public void setAlpha(double newAlpha) {
	alpha = newAlpha;
    }

}