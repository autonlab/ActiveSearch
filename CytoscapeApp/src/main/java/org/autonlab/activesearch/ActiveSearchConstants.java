package org.autonlab.activesearch;

public class ActiveSearchConstants {

    public static final String REST_URL_PREFIX = "http://localhost:8080/ActiveSearchDaemon/rest/";

    /* The JSlider class can only work with ints so we'll divide this by 1000 when we actually apply it */
    public static final int SEARCH_MAIN_ALPHA_INIT = 0;
    public static final int SEARCH_MAIN_ALPHA_MIN = 0;
    public static final int SEARCH_MAIN_ALPHA_MAX = 10;

    public static final int HIDE_SEED_FIELD = -1;

    public static final int MODE_DO_NOTHING = -1;
    public static final int MODE_DO_NOTHING_USER_WILL_PICK_SEED = -2;
    public static final int MODE_START_RANDOM = 0;
    public static final int MODE_SHOW_SELECT_SEED = 1;
    public static final int MODE_SHOW_LAST_SEED = 2;
    public static final int MODE_SHOW_EDGE_EMAILS = 3;
}