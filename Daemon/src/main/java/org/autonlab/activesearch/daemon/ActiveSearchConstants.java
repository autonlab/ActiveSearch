package org.autonlab.activesearch;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.regex.*;

public class ActiveSearchConstants {

    public static String DATABASE_NAME = "scottwalker";
    public static int DATABASE_PORT = 3306;
    public static String DATABASE_USERNAME = "root";
    public static String DATABASE_PASSWORD = "";

    public static String SIMILARITY_MATRIX_FILE = "similarity_rowsum.out";
    public static String X_MATRIX = "X_matrix.out";
    public static String b_MATRIX = "b_matrix.out";

    // by default, this is generated automatically so we don't set it
    public static String LABELS_FILE = "";

    public static int SEARCH_MAIN_DIMENSIONS = 2000;

    /* The JSlider class can only work with ints so we'll divide this by 1000 when we actually apply it */
    public static int SEARCH_MAIN_ALPHA_INIT = 0;
    public static int SEARCH_MAIN_ALPHA_MIN = 0;
    public static int SEARCH_MAIN_ALPHA_MAX = 10;

    public static double SEARCH_MAIN_OMEGA = -1;

    public static int HIDE_SEED_FIELD = -1;

    /* these are states in a state machine so we don't let the user configure them */
    public static final int MODE_DO_NOTHING = -1;
    public static final int MODE_DO_NOTHING_USER_WILL_PICK_SEED = -2;
    public static final int MODE_START_RANDOM = 0;
    public static final int MODE_SHOW_SELECT_SEED = 1;
    public static final int MODE_SHOW_LAST_SEED = 2;
    public static final int MODE_SHOW_EDGE_EMAILS = 3;

    public static void readConfig(String configFile) {
	BufferedReader br = null;
	String sCurrentLine;

	try {

	    br = new BufferedReader(new FileReader(configFile));
	    System.out.println("Importing config");

	    while ((sCurrentLine = br.readLine()) != null) {
		// skip blank lines and lines that start with a comment (#)
		if (sCurrentLine.matches("^\\s*$") ||
		    sCurrentLine.matches("^\\s*#.*$")) {
		    System.out.println("skipping this line: " + sCurrentLine);
		    continue;
		}
		Pattern pattern = Pattern.compile("^\\s*(\\w+)\\s*=\\s*([\\w\\/]+)");
		Matcher matcher = pattern.matcher(sCurrentLine);
		if (matcher.find()) {
		    String key = matcher.group(1);
		    String val = matcher.group(2);

		    System.out.println("Setting key " + key + " to " + val);
		    if (key.equalsIgnoreCase("DATABASE_NAME")) {
			DATABASE_NAME = val;
		    }
		    else if (key.equalsIgnoreCase("DATABASE_PORT")) {
			DATABASE_PORT = Integer.parseInt(val);
		    }
		    else if (key.equalsIgnoreCase("DATABASE_USERNAME")) {
			DATABASE_USERNAME = val;
		    }
		    else if (key.equalsIgnoreCase("DATABASE_PASSWORD")) {
			DATABASE_PASSWORD = val;
		    }
		    else if (key.equalsIgnoreCase("SIMILARITY_MATRIX_FILE")) {
			SIMILARITY_MATRIX_FILE = val;
		    }
		    else if (key.equalsIgnoreCase("X_MATRIX")) {
			X_MATRIX = val;
		    }
		    else if (key.equalsIgnoreCase("b_MATRIX")) {
			b_MATRIX = val;
		    }
		    else if (key.equalsIgnoreCase("LABELS_FILE")) {
			LABELS_FILE = val;
		    }
		    else if (key.equalsIgnoreCase("SEARCH_MAIN_DIMENSIONS")) {
			SEARCH_MAIN_DIMENSIONS = Integer.parseInt(val);
		    }
		    else if (key.equalsIgnoreCase("SEARCH_MAIN_ALPHA_INIT")) {
			SEARCH_MAIN_ALPHA_INIT = Integer.parseInt(val);
		    }
		    else if (key.equalsIgnoreCase("SEARCH_MAIN_ALPHA_MIN")) {
			SEARCH_MAIN_ALPHA_MIN = Integer.parseInt(val);
		    }
		    else if (key.equalsIgnoreCase("SEARCH_MAIN_ALPHA_MAX")) {
			SEARCH_MAIN_ALPHA_MAX = Integer.parseInt(val);
		    }
		    else if (key.equalsIgnoreCase("SEARCH_MAIN_OMEGA")) {
			SEARCH_MAIN_OMEGA = Double.parseDouble(val);
		    }
		    else if (key.equalsIgnoreCase("HIDE_SEED_FIELD")) {
			HIDE_SEED_FIELD = Integer.parseInt(val);
		    }
		    else {
			throw new RuntimeException("No config key " + key);
		    }
		}
	    }
	} catch (IOException e) {
	    e.printStackTrace();
	    throw new RuntimeException("Failed to read file " + configFile);
	} finally {
	    try {
		if (br != null)br.close();
	    } catch (IOException ex) {
		ex.printStackTrace();
		throw new RuntimeException("Failed to read file " + configFile);
	    }
	}
    }
}
