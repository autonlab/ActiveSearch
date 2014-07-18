package org.autonlab.activesearch;

public class ActiveSearchConstants {

    public static final String DATABASE_NAME = "scottwalker";
    public static final int DATABASE_PORT = 3306;
    public static final String DATABASE_USERNAME = "root";
    public static final String DATABASE_PASSWORD = "";

    public static final String SIMILARITY_MATRIX_FILE = "similarity_rowsum.out";
    public static final String X_MATRIX = "X_matrix.out";
    public static final String b_MATRIX = "b_matrix.out";

    // by default, this is generated automatically so we don't set it
    public static final String LABELS_FILE = "";

    public static final int SEARCH_MAIN_DIMENSIONS = 2000;

    /* The JSlider class can only work with ints so we'll divide this by 1000 when we actually apply it */
    public static final int SEARCH_MAIN_ALPHA_INIT = 0;
    public static final int SEARCH_MAIN_ALPHA_MIN = 0;
    public static final int SEARCH_MAIN_ALPHA_MAX = 10;

    public static final double SEARCH_MAIN_OMEGA = -1;
    public static final int SEARCH_MAIN_OFFSET_FLAG = 1;

    public static final int HIDE_SEED_FIELD = -1;

    public static final int MODE_DO_NOTHING = -1;
    public static final int MODE_DO_NOTHING_USER_WILL_PICK_SEED = -2;
    public static final int MODE_START_RANDOM = 0;
    public static final int MODE_SHOW_SELECT_SEED = 1;
    public static final int MODE_SHOW_LAST_SEED = 2;
    public static final int MODE_SHOW_EDGE_EMAILS = 3;
}
