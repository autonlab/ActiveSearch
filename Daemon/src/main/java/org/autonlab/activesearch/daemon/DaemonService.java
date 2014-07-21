package org.autonlab.activesearch.daemon;
 
import javax.ws.rs.*;
import javax.ws.rs.core.*;
import org.jblas.*;
import org.autonlab.activesearch.ActiveSearchConstants;
import org.autonlab.activesearch.DataConnectionMySQL;
import org.autonlab.activesearch.GenerateCompabilityMatrix;
import org.autonlab.activesearch.SearchMain;
import org.autonlab.activesearch.EmailSimilarity;
import org.autonlab.activesearch.GlapEigenmap;

@Path("/")
public class DaemonService {
 
    static DoubleMatrix xmappedMatrix = null;
    static int nConnComp = 0;
    static DoubleMatrix similarityMatrix = null;
    static DoubleMatrix labelsMatrixOrig = null;
    static DoubleMatrix labelsMatrix = null;
    static SearchMain aSearch = null;
    static int currentEmail;
    static DataConnectionMySQL dataConnection = null;

    /* dividing by 1000 is an artifact of the CytoscapeApp where the slider class could only operate on ints */
    static double alpha = ActiveSearchConstants.SEARCH_MAIN_ALPHA_INIT / 1000.0;

    public void initState() {
	System.out.println("Initializing state. This will take a minute and only needs to be done once.");
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}
	int emailCount = dataConnection.getTotalEmailCount();
	if (similarityMatrix == null) {
	    similarityMatrix = GenerateCompabilityMatrix.readFile(ActiveSearchConstants.SIMILARITY_MATRIX_FILE, emailCount, 1);
	    xmappedMatrix = GenerateCompabilityMatrix.readFile(ActiveSearchConstants.X_MATRIX, emailCount, 0);
	    nConnComp = (int)(GenerateCompabilityMatrix.readFile(ActiveSearchConstants.b_MATRIX, 1, 0).get(0));
	    System.out.println("nConnComp is " + nConnComp);
	    if ((ActiveSearchConstants.LABELS_FILE).equals("")) {
		System.out.println("No labels file defined. Using default vector");
		labelsMatrixOrig = DoubleMatrix.zeros(emailCount);
	    }
	    else {
		labelsMatrixOrig = GenerateCompabilityMatrix.readFile(ActiveSearchConstants.LABELS_FILE, emailCount, 1);
	    }
	}
	System.out.println("Matrices loaded");
    }
 
    /*
     * Calling this resets the labels
     */
    @GET
    @Path("/firstemail/{email}/{mode}")
    public Response firstEmail(@PathParam("email") int email,
			       @PathParam("mode") int mode) {
	if (similarityMatrix == null) {
	    initState();
	}
	labelsMatrix = labelsMatrixOrig.dup();

	aSearch = new SearchMain(xmappedMatrix, similarityMatrix, 
				 ActiveSearchConstants.SEARCH_MAIN_DIMENSIONS, 
				 (int)(alpha * 1000),
				 ActiveSearchConstants.SEARCH_MAIN_OMEGA,
				 email,
				 ActiveSearchConstants.SEARCH_MAIN_OFFSET_FLAG,
				 labelsMatrix,
				 mode,
				 nConnComp);
	System.out.println("First email is " + email + " in mode " + mode);
	String output = "ok";
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/firstemail/{email}")
    public Response firstEmail(@PathParam("email") int email) {
	return firstEmail(email, ActiveSearchConstants.MODE_DO_NOTHING_USER_WILL_PICK_SEED);
    }


    @GET
    @Path("/emailinteresting")
    public Response interestingEmail() {
	aSearch.setLabel(currentEmail, 1);
	currentEmail = aSearch.getNextEmail();
	String output = "" + currentEmail;
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/emailboring")
    public Response boringEmail() {
	aSearch.setLabel(currentEmail, 0);
	currentEmail = aSearch.getNextEmail();
	String output = "" + currentEmail;
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/setalpha/{alpha}")
    public Response setAlpha(@PathParam("alpha") double newAlpha) {
	if (similarityMatrix == null) {
	    initState();
	}

	if (aSearch != null) {
	    aSearch.setAlpha(newAlpha);
	}
	else {
	    alpha = newAlpha;
	}
	System.out.println("Alpha set to " + newAlpha);

	String output = "ok";
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getStartPoint")
    public Response getStartPoint() {
	int startPoint = aSearch.getStartPoint();
	String output = "" + startPoint;
	System.out.println("Retrieved start point of " + startPoint);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/resetLabel/{index}/{value}")
    public Response resetLabel(@PathParam("index") int index,
			       @PathParam("value") int value) {
	double oldVal = aSearch.resetLabel(index, value);
	System.out.println("Label for " + index + " changed from " + oldVal + " to " + value);
	String output = "" + oldVal;
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/setLabelCurrent/{value}")
    public Response setLabelCurrent(@PathParam("value") int value) {
	aSearch.setLabel(currentEmail, value);
	System.out.println("Label for " + currentEmail + " set to " + value);
	String output = "ok";
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/setLabel/{index}/{value}")
    public Response setLabel(@PathParam("index") int index,
			     @PathParam("value") int value) {
	aSearch.setLabel(index, value);
	System.out.println("Label for " + index + " set to " + value);
	String output = "ok";
	return Response.status(200).entity(output).build();
    }

    /*
     * Input is [index, value [,index, value etc]]
     */
    @GET
    @Path("/setLabelBulk/{csv}")
    public Response setLabelBulk(@PathParam("csv") String csv) {
	String output = "ok";
	String[] sParts;
	int i;
	sParts = csv.split(",");
	if (sParts.length % 2 != 0) {
	    output = "Error: odd number of inputs";
	}
	else {
	    for (i = 0; i < sParts.length; i+=2) {
		int index = Integer.parseInt(sParts[i]);
		double value = Double.parseDouble(sParts[i+1]);
		aSearch.setLabel(index, value);
		System.out.println("Label for " + index + " set to " + value);
	    }
	}

	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getNextEmail")
    public Response getNextEmail() {
	currentEmail = aSearch.getNextEmail();
	String output = "" + currentEmail;
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/pickRandomLabeledEmail")
    public Response pickRandomLabeledEmail() {
	int email = aSearch.pickRandomLabeledEmail();
	String output = "" + email;
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getLabel/{email}")
    public Response getLabel(@PathParam("email") int email) {
	/* The label is read in when the active search starts processing but we display emails before that so we have to hide the field, hence the -1.0*/
	double label = -1.0;
	if (aSearch != null) {
	    label = aSearch.getLabel(email);
	}
	String output = "" + label;
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/eigenmap/{count}")
    public Response generateEigenmap(@PathParam("count") int threadCount) {
	int i;   
	EmailSimilarity[] emailData;

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

	
	String output = "Done generating Eigenmap";
	return Response.status(200).entity(output).build();
    }

    /*
     * This section implements all of the functions of DataConnectionMySQL so we can separate out the UI
     */

    @GET
    @Path("/getUserNameFromID/{id}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getUserNameFromID(@PathParam("id") int id) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getUserNameFromID(id);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getMessagesFromUserToUser/{fromID}/{toID}")
    @Produces(MediaType.TEXT_PLAIN)
	public Response getMessagesFromUserToUser(@PathParam("fromID") int userIDFrom, 
						  @PathParam("toID") int userIDTo) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getMessagesFromUserToUser(userIDFrom, userIDTo);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getEmailSubjectFromMessageID/{id}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getEmailSubjectFromMessageID(@PathParam("id") int id) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getEmailSubjectFromMessageID(id);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getEmailBodyFromMessageID/{id}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getEmailBodyFromMessageID(@PathParam("id") int id) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getEmailBodyFromMessageID(id);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getTotalEmailCount")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getEmailBodyFromMessageID() {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = "" + dataConnection.getTotalEmailCount();
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getEmailTimesAndSenders")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getEmailTimesAndSenders() {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = "" + dataConnection.getEmailTimesAndSenders();
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getUsersByEmail/{id}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getUsersByEmail(@PathParam("id") int id) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = "" + dataConnection.getUsersByEmail(id);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getSenderByEmail/{id}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getSenderByEmail(@PathParam("id") int id) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = "" + dataConnection.getSenderByEmail(id);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getTimeByEmail/{id}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getTimeByEmail(@PathParam("id") int id) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getTimeByEmail(id);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getSubjectByEmail/{id}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getSubjectByEmail(@PathParam("id") int id) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getSubjectByEmail(id);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getEmailsByKeyword/{word}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getEmailsByKeyword(@PathParam("word") String word) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getEmailsByKeyword(word);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/getEmailsByKeywordSubject/{word}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getEmailsByKeywordSubject(@PathParam("word") String word) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getEmailsByKeywordSubject(word);
	return Response.status(200).entity(output).build();
    }



    /*
     * we don't implement DataConnectionMySQL.getEmailRecipients() because
     * the caller can just loop over getEmailRecipientsByEmail from 0 to emailCount-1
     */


    @GET
    @Path("/getEmailRecipientsByEmail/{email}")
    @Produces(MediaType.TEXT_PLAIN)
    public Response getEmailRecipientsByEmail(@PathParam("email") int email) {
	if (dataConnection == null) {
	    dataConnection = new DataConnectionMySQL();
	}

	String output = dataConnection.getEmailRecipientsByEmail(email);
	return Response.status(200).entity(output).build();
    }

    @GET
    @Path("/readConfigFile")
    @Produces(MediaType.TEXT_PLAIN)
    public Response readConfigFile(@QueryParam("configfile") String configFile) {
	ActiveSearchConstants.readConfig(configFile);

	String output = "ok";
	return Response.status(200).entity(output).build();
    }



}
