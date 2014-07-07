package org.autonlab.activesearch.ui;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.FlowLayout;
import java.awt.Point;

import javax.swing.*;
import javax.swing.event.*;
import java.awt.Component;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.*;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.autonlab.activesearch.ActiveSearchConstants;
import org.autonlab.activesearch.DataConnectionRest;
import org.autonlab.activesearch.tasks.UpdateNodeNameViewTask;
import org.cytoscape.model.CyEdge.Type;
import org.cytoscape.application.CyApplicationManager;

import java.util.LinkedList;
import java.util.TreeMap;
import java.util.List;
import java.util.Iterator;
import java.util.Collection;

import java.awt.Color;
import java.awt.Paint;

import org.jblas.DoubleMatrix;
import org.cytoscape.view.presentation.property.BasicVisualLexicon;
import org.autonlab.activesearch.tasks.ShowFullEmailEdgeViewTask;
import org.autonlab.activesearch.ui.StatisticsDialog;
import org.autonlab.activesearch.ActiveSearchConstants;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class FullEmailDialog extends JDialog implements ActionListener{

    // JDialog requires this
    private static final long serialVersionUID = 1L;

    View<CyEdge> edgeView = null;
    CyNetworkView netView;
    JTextArea areaEmailContents;
    DataConnectionRest dataConnectionHandle;

    JButton buttonEmailInteresting, buttonEmailBoring;
    JScrollPane panelEmailSubjectsScrollpane, panelEmailBodyScrollpane;
    JPanel panelEmailSubjects;
    JPanel panelEmailBody;
    JPanel panelBottomVote;
    JLabel labelSender;
    JLabel labelReceivers;
    JLabel labelSubject;
    JLabel labelTime;
    JLabel labelASLabel;
    JSlider sliderAlpha;

    int displayedEmail = -1;
    int displayedEmailSubject = -1;
    GroupLayout layout;
    CySwingAppAdapter adapter;
    GroupLayout.SequentialGroup sGroup;

    int mode;

    StatisticsDialog statsDialog;

    FullEmailDialog mySelf;

    int nextEmailIsRevote;
    int emailIsInitialized = 0;

    public FullEmailDialog(CyNetworkView netView, View<CyEdge> edgeView, CySwingAppAdapter myAdapter, int myMode)
    {
	super();
	this.edgeView = edgeView;
	this.netView = netView;
	adapter = myAdapter;
	mode = myMode;
	nextEmailIsRevote = 0;
	mySelf = this;

	dataConnectionHandle = new DataConnectionRest();

	int emailCount = dataConnectionHandle.getTotalEmailCount();

	UpdateNodeNameViewTask.UpdateNodeNames(netView);

	/* the top panel where the email subjects are displayed */
	panelEmailSubjects = new JPanel();
	panelEmailSubjects.setOpaque(true);
 	panelEmailSubjects.setLayout(new BoxLayout(panelEmailSubjects, BoxLayout.Y_AXIS));
	panelEmailSubjectsScrollpane = new JScrollPane(panelEmailSubjects, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
						  JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	panelEmailSubjectsScrollpane.setPreferredSize(new Dimension(800,600));

	/* the bottom text area where the email contents are displayed */
	panelEmailBody = new JPanel();
	panelEmailBody.setOpaque(true);
	panelEmailBody.setLayout(new BoxLayout(panelEmailBody, BoxLayout.Y_AXIS));

	panelEmailBodyScrollpane = new JScrollPane(panelEmailBody, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
						  JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	panelEmailBodyScrollpane.setPreferredSize(new Dimension(800,600));

	labelSender = new JLabel();
	labelSender.setAlignmentX(Component.LEFT_ALIGNMENT);
	panelEmailBody.add(labelSender);
	labelReceivers = new JLabel();
	labelReceivers.setAlignmentX(Component.LEFT_ALIGNMENT);
	panelEmailBody.add(labelReceivers);
	labelSubject = new JLabel();
	labelSubject.setAlignmentX(Component.LEFT_ALIGNMENT);
	panelEmailBody.add(labelSubject);
	labelTime = new JLabel();
	labelTime.setAlignmentX(Component.LEFT_ALIGNMENT);
	panelEmailBody.add(labelTime);
	labelASLabel = new JLabel();
	labelASLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
	panelEmailBody.add(labelASLabel);
	areaEmailContents = new JTextArea();
	areaEmailContents.setEditable(false);
	areaEmailContents.setLineWrap(true);
	areaEmailContents.setAlignmentY(TOP_ALIGNMENT);

	panelEmailBody.add(areaEmailContents);
	areaEmailContents.setAlignmentX(Component.LEFT_ALIGNMENT);

	buttonEmailInteresting = new JButton("Interesting");
	buttonEmailInteresting.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent event){
		    if (displayedEmailSubject != -1) {
			displayedEmail = displayedEmailSubject;
			displayedEmailSubject = -1;
		    }

		    if (displayedEmail < 0) {
			throw new RuntimeException("Tried to tag undefined email as Interesting");
		    }

		    if (mode == ActiveSearchConstants.MODE_START_RANDOM) {
			buttonEmailInteresting.setVisible(true);
			buttonEmailBoring.setVisible(true);
			if (nextEmailIsRevote == 1) {
			    double oldVote = dataConnectionHandle.resetLabel(displayedEmail, 1);
			    statsDialog.addEmail(displayedEmail, 1, nextEmailIsRevote, (int)oldVote);
			    nextEmailIsRevote = 0;
			}
			else {
			    dataConnectionHandle.setLabel(1);
			    statsDialog.addEmail(displayedEmail, 1, nextEmailIsRevote, -1);
			}
			hideDistantEmails(dataConnectionHandle.getNextEmail());
		    }
		    else if (mode == ActiveSearchConstants.MODE_SHOW_SELECT_SEED | mode == ActiveSearchConstants.MODE_SHOW_LAST_SEED) {
			buttonEmailInteresting.setText("Interesting");
			buttonEmailBoring.setText("Boring");

			buttonEmailInteresting.setVisible(true);
			buttonEmailBoring.setVisible(true);

			// in these modes, aSearch was created earlier but the values are invalid
			// create a new SearchMain with correct values now that we know the appropriate seed email
			dataConnectionHandle.firstEmail(displayedEmail, ActiveSearchConstants.MODE_START_RANDOM);
			emailIsInitialized = 1;
			mode = ActiveSearchConstants.MODE_START_RANDOM;

			hideDistantEmails(dataConnectionHandle.getNextEmail());

			if (statsDialog != null) {
			    statsDialog.showSeed();
			}
		    }
		    else if (mode == ActiveSearchConstants.MODE_SHOW_EDGE_EMAILS ||
			     mode == ActiveSearchConstants.MODE_DO_NOTHING_USER_WILL_PICK_SEED) {
			System.out.println("Initializing yes with " + displayedEmail);
			dataConnectionHandle.firstEmail(displayedEmail, mode);
			emailIsInitialized = 1;
			statsDialog = new StatisticsDialog(dataConnectionHandle, mySelf);

			// this is here so we pass MODE_SHOW_EDGE_EMAILS to SearchMain() but that we've remapped it to MODE_START_RANDOM
			// before we go into hideDistantEmails
			mode = ActiveSearchConstants.MODE_START_RANDOM;

			buttonEmailInteresting.setVisible(true);
			buttonEmailBoring.setVisible(true);
			hideDistantEmails(dataConnectionHandle.getNextEmail());
		    }
		    else {
			throw new RuntimeException("Mode " + mode + " not implemented");
		    }
		}
	    });

	buttonEmailBoring = new JButton("Boring");
	buttonEmailBoring.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent event){
		    if (displayedEmailSubject != -1) {
			displayedEmail = displayedEmailSubject;
			displayedEmailSubject = -1;
		    }

		    if (displayedEmail < 0) {
			throw new RuntimeException("Tried to tag undefined email as Uninteresting");
		    }

		    if (mode == ActiveSearchConstants.MODE_START_RANDOM) {
			buttonEmailInteresting.setVisible(true);
			buttonEmailBoring.setVisible(true);

			if (emailIsInitialized == 0) {
			    // this can happen if we're coming from MODE_SHOW_EDGE_EMAILS where we display the subjects and the user picks
			    System.out.println("Initializing no with " + displayedEmail);
			    dataConnectionHandle.firstEmail(displayedEmail, mode);
			    emailIsInitialized = 1;
			}

			if (nextEmailIsRevote == 1) {
			    double oldVote = dataConnectionHandle.resetLabel(displayedEmail, 0);
			    statsDialog.addEmail(displayedEmail, 0, nextEmailIsRevote, (int)oldVote);
			    nextEmailIsRevote = 0;
			}
			else {
			    dataConnectionHandle.setLabel(0);
			    statsDialog.addEmail(displayedEmail, 0, nextEmailIsRevote, -1);
			}
			hideDistantEmails(dataConnectionHandle.getNextEmail());
		    }
		    else if (mode == ActiveSearchConstants.MODE_SHOW_SELECT_SEED) {
			buttonEmailInteresting.setVisible(true);
			buttonEmailBoring.setVisible(true);
			hideDistantEmails(dataConnectionHandle.pickRandomLabeledEmail());
		    }
		    else if (mode == ActiveSearchConstants.MODE_SHOW_LAST_SEED) {
			throw new RuntimeException("Should not have been able to click boring in mode 2");
		    }
		    else {
			throw new RuntimeException("Mode " + mode + " not implemented");
		    }
		}
	    });

	sliderAlpha = new JSlider(JSlider.HORIZONTAL, ActiveSearchConstants.SEARCH_MAIN_ALPHA_MIN, ActiveSearchConstants.SEARCH_MAIN_ALPHA_MAX, ActiveSearchConstants.SEARCH_MAIN_ALPHA_INIT);
	sliderAlpha.addChangeListener(new ChangeListener() {
		public void stateChanged(ChangeEvent event) {
		     JSlider source = (JSlider)event.getSource();
		     if (!source.getValueIsAdjusting()) {
			 int sliderValue = source.getValue();
			 dataConnectionHandle.setAlpha(((double)sliderValue) / 1000.0);
		     }
		}
	    });
	sliderAlpha.setMajorTickSpacing(10);
	sliderAlpha.setMinorTickSpacing(1);
	sliderAlpha.setPaintTicks(true);
	sliderAlpha.setPaintLabels(true);

	/* the panel at the very bottom that contains the interesting/boring buttons */
	panelBottomVote = new JPanel();
	panelBottomVote.setOpaque(true);
	panelBottomVote.setLayout(new FlowLayout());
	panelBottomVote.add(buttonEmailInteresting);
	panelBottomVote.add(buttonEmailBoring);
	panelBottomVote.add(sliderAlpha);

	layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setAutoCreateGaps(false);
        layout.setAutoCreateContainerGaps(true);
 	
	GroupLayout.ParallelGroup pGroup = layout.createParallelGroup(GroupLayout.Alignment.LEADING);
	pGroup.addComponent(panelEmailSubjectsScrollpane);
	pGroup.addComponent(panelEmailBodyScrollpane);
	pGroup.addComponent(panelBottomVote);
	layout.setHorizontalGroup(layout.createSequentialGroup().addGroup(pGroup));

	sGroup = layout.createSequentialGroup();
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(panelEmailSubjectsScrollpane));
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(panelEmailBodyScrollpane));
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(panelBottomVote));
        layout.setVerticalGroup(sGroup);

	pack();
	setAlwaysOnTop(false);
	setResizable(true);
	setLocationRelativeTo(null);
	setVisible(true);
	setLocation(10, 10);

	int showEmail = -1;
	if (mode == ActiveSearchConstants.MODE_START_RANDOM) {
	    dataConnectionHandle.firstEmail(-1, mode);
	    emailIsInitialized = 1;
	    showEmail = dataConnectionHandle.getNextEmail();
	    hideDistantEmails(showEmail);
	}
	else if (mode == ActiveSearchConstants.MODE_SHOW_SELECT_SEED) {
	    dataConnectionHandle.firstEmail(-1, mode);
	    emailIsInitialized = 1;
	    showEmail = dataConnectionHandle.getStartPoint();
	    hideDistantEmails(showEmail);
	}
	else if (mode == ActiveSearchConstants.MODE_SHOW_LAST_SEED) {
	    hideDistantEmails(dataConnectionHandle.getStartPoint());
	}
	else if (mode == ActiveSearchConstants.MODE_SHOW_EDGE_EMAILS) {
	    CyNetwork myNetwork = netView.getModel();

	    CyNode sourceNode = edgeView.getModel().getSource();
	    // The nodes sometimes are named like "8" _with_ the quotes so we need to drop the quotes before converting to an int
	    int sourceNodeID = Integer.parseInt(myNetwork.getRow(sourceNode).get(CyNetwork.NAME, String.class).replaceAll("\"", ""));
	    String sourceNodeUser = "";
	    CyNode targetNode = edgeView.getModel().getTarget();
	    int targetNodeID = Integer.parseInt(myNetwork.getRow(targetNode).get(CyNetwork.NAME, String.class).replaceAll("\"", ""));
	    String targetNodeUser = "";

	    sourceNodeUser = dataConnectionHandle.getUserNameFromID(sourceNodeID);
	    targetNodeUser = dataConnectionHandle.getUserNameFromID(targetNodeID);
	    setTitle("Emails from " + sourceNodeUser + " to " + targetNodeUser);

	    //	    DisplayFullSubjectDialog(dataConnectionHandle.getMessagesFromUserToUser(sourceNodeID, targetNodeID));
	    DisplayFullSubjectDialog(dataConnectionHandle.getMessagesFromUserToUser(sourceNodeID, targetNodeID));
	}
	else if (mode == ActiveSearchConstants.MODE_DO_NOTHING ||
		 mode == ActiveSearchConstants.MODE_DO_NOTHING_USER_WILL_PICK_SEED) {
	    // do nothing, but have this case so we don't fall into the error condition
	}
	else {
	    throw new RuntimeException("Mode " + mode + " not implemented");
	}

	if (displayedEmail != -1) {
	    if (mode == ActiveSearchConstants.MODE_SHOW_SELECT_SEED) {
		statsDialog = new StatisticsDialog(dataConnectionHandle, mySelf, ActiveSearchConstants.HIDE_SEED_FIELD);
	    }
	    else {
		statsDialog = new StatisticsDialog(dataConnectionHandle, mySelf);
	    }
	}
    }

    public void DisplayFullSubjectDialog(LinkedList<String> subjectListString) {
	String emailText = "Please select an email above to view its contents";

	displayedEmail = -1;
	displayedEmailSubject = -1;
	panelEmailSubjects.removeAll();

	panelEmailSubjectsScrollpane.setVisible(true);
	labelSender.setVisible(false);
	labelReceivers.setVisible(false);
	labelTime.setVisible(false);
	labelSubject.setVisible(false);
	labelASLabel.setVisible(false);
	panelEmailBodyScrollpane.setVisible(false);
	panelEmailBody.setVisible(false);

	buttonEmailInteresting.setText("Start Making Recommendations");
	buttonEmailInteresting.setVisible(false);
	buttonEmailBoring.setText("");
	buttonEmailBoring.setVisible(false);

	sliderAlpha.setVisible(false);

	for (String b : subjectListString) {
	    JButton temp = new JButton();
	    temp.setText(b);
	    temp.addActionListener(this);
	    panelEmailSubjects.add(temp);
	}

	areaEmailContents.setText(emailText);
	areaEmailContents.setCaretPosition(0);

	panelEmailSubjects.revalidate();
	panelEmailSubjects.repaint();

	revalidate();
	repaint();
	netView.updateView();
    }

    /**
     * Populate the popup window with email information
     *
     * @param email Email ID to display, or -1 to display all email subjects between two users rather than a specific email
     */
    public void DisplayFullEmailDialog(int email) {
	if (email < 0) {
	    throw new RuntimeException("email ID was not valid: " + email);
	}

	displayedEmail = email;

	panelEmailSubjects.removeAll();

	String emailText;


	emailText = dataConnectionHandle.getEmailBodyFromMessageID(displayedEmail);

	if (mode == ActiveSearchConstants.MODE_START_RANDOM) {
	    buttonEmailInteresting.setText("Interesting");
	    buttonEmailBoring.setText("Boring");
	}
	else if (mode == ActiveSearchConstants.MODE_SHOW_SELECT_SEED) {
	    buttonEmailInteresting.setText("Start Making Recommendations");
	    buttonEmailBoring.setText("Show Next Positive Email");
	}
	else if (mode == ActiveSearchConstants.MODE_SHOW_LAST_SEED) {
	    buttonEmailInteresting.setText("Start Making Recommendations");
	    buttonEmailBoring.setText("");
	    buttonEmailBoring.setVisible(false);
	    sliderAlpha.setVisible(false);
	}
	else {
	    throw new RuntimeException("Mode " + mode + " not implemented");
	}

	panelEmailSubjectsScrollpane.setVisible(false);
	showEmailBody(email);

	areaEmailContents.setText(emailText);
	areaEmailContents.setCaretPosition(0);

	panelEmailSubjects.revalidate();
	panelEmailSubjects.repaint();

	revalidate();
	repaint();
	netView.updateView();
    }

    public void showEmailBody(int email) {
	buttonEmailInteresting.setVisible(true);
	if (mode == ActiveSearchConstants.MODE_SHOW_LAST_SEED ||
	    mode == ActiveSearchConstants.MODE_SHOW_EDGE_EMAILS||
	    mode == ActiveSearchConstants.MODE_DO_NOTHING_USER_WILL_PICK_SEED) {
	    buttonEmailBoring.setVisible(false);
	    sliderAlpha.setVisible(false);
	}
	else {
	    buttonEmailBoring.setVisible(true);
	    sliderAlpha.setVisible(true);
	    buttonEmailInteresting.setVisible(true);
	}

	/* get the list of recipient IDs, convert those to names, then sort the names */
	LinkedList<String> listRecipientsNames = new LinkedList<String>();
	LinkedList<Integer> listRecipients = dataConnectionHandle.getEmailRecipientsByEmail(email);
	Iterator<Integer> listRecipientsIter = listRecipients.iterator();
	while (listRecipientsIter.hasNext()) {
	    Integer myInt = listRecipientsIter.next();
	    listRecipientsNames.add(dataConnectionHandle.getUserNameFromID(myInt));
	}
	java.util.Collections.sort(listRecipientsNames, String.CASE_INSENSITIVE_ORDER);
	Iterator<String> listRecipientsNamesIter = listRecipientsNames.iterator();
	labelReceivers.setText("<html>Recipients:");
	while (listRecipientsNamesIter.hasNext()) {
	    String myName = listRecipientsNamesIter.next();
	    labelReceivers.setText(labelReceivers.getText() + "<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + myName);
	}
	labelReceivers.setText(labelReceivers.getText() + "</html>");

	labelSender.setText("Sender: " + dataConnectionHandle.getUserNameFromID(dataConnectionHandle.getSenderByEmail(email)));
	labelTime.setText("Time: " + dataConnectionHandle.getTimeByEmail(email));
	labelSubject.setText("Subject: " + dataConnectionHandle.getSubjectByEmail(email));
	/* The label is read in when the active search starts processing but we display emails before that so we have to hide the field */
	double labelVal = dataConnectionHandle.getLabel(email);
	if (labelVal >= 0.0) {
	    labelASLabel.setText("Label: " + labelVal);
	    labelASLabel.setVisible(true);
	}
	else {
	    labelASLabel.setVisible(false);
	}

	labelSender.setVisible(true);
	labelReceivers.setVisible(true);
	labelTime.setVisible(true);
	labelSubject.setVisible(true);

	panelEmailBody.setVisible(true);
	panelEmailBodyScrollpane.setVisible(true);
	setTitle("Recommendations Window. Showing Email ID " + email);

	panelEmailSubjects.revalidate();
	panelEmailSubjects.repaint();

	revalidate();
	repaint();
	netView.updateView();
    }

    /*
     * This is called when email subjects are displayed and a single email is selected 
     */
    public void actionPerformed(ActionEvent e) {
	String emailText;

	JButton source = (JButton)e.getSource();

	// source text is the form <message id> : <timestamp> : <subject> so split it to get just the messageid
	String[] parts = source.getText().split(" : ");
	int email = Integer.parseInt(parts[0]);

	displayedEmailSubject = email;

	showEmailBody(email);

	emailText = dataConnectionHandle.getEmailBodyFromMessageID(email);
	areaEmailContents.setText(emailText);
	areaEmailContents.setCaretPosition(0);

	panelEmailSubjects.revalidate();
	panelEmailSubjects.repaint();

	revalidate();
	repaint();
	netView.updateView();
    }

    public void setNextEmailIsReVote() {
	nextEmailIsRevote = 1;
    }

    /**
     * Get an email's sender and receivers and a list of users adjacent to them. Hide all other nodes
     */
    public void hideDistantEmails(int email) {
	Double nodeX = -1.0;
	Double nodeY = -1.0;

	CyApplicationManager manager = adapter.getCyApplicationManager();
	CyNetworkView networkView = manager.getCurrentNetworkView();
	CyNetwork myNetwork = networkView.getModel();

	TreeMap<Long, CyNode> nodesToKeep = new TreeMap<Long, CyNode>();

	LinkedList<Integer> adjacentNodes = dataConnectionHandle.getUsersByEmail(email);
	Iterator<Integer> adjacentNodesIter = adjacentNodes.iterator();

	CyTable table = myNetwork.getDefaultNodeTable();

	System.out.println("Next email is " + email);

	while (adjacentNodesIter.hasNext()) {
	    Integer myInt = adjacentNodesIter.next();

	    // translate from a userID to CyNodes for those users
	    // http://wiki.cytoscape.org/Cytoscape_3/AppDeveloper/Cytoscape_3_App_Cookbook#How_to_get_all_the_nodes_with_a_specific_attribute_value.3F
	    Collection<CyRow> matchingRows = table.getMatchingRows(CyNetwork.NAME, myInt.toString());
	    String primaryKeyColname = table.getPrimaryKey().getName();
	    CyNode node = null;
	    for (CyRow row : matchingRows) {
		Long nodeId = row.get(primaryKeyColname, Long.class);
		if (nodeId == null) {
		    continue;
		}
		node = myNetwork.getNode(nodeId);
		if (node == null) {
		    continue;
		}
		break;
	    }

	    if (node != null) {
		if (!nodesToKeep.containsKey(node.getSUID())) {
		    nodesToKeep.put(node.getSUID(),node);
		}

		if (nodeX < 0.0) {
		    nodeX = networkView.getNodeView(node).getVisualProperty(BasicVisualLexicon.NODE_X_LOCATION);
		    nodeY = networkView.getNodeView(node).getVisualProperty(BasicVisualLexicon.NODE_Y_LOCATION);
		}

		/*
		  // This will include people one hop away but there are so many that it isn't too interesting
		List<CyNode> myNeighbors = myNetwork.getNeighborList(node, CyEdge.Type.ANY);
		Iterator<CyNode> myNeighborsIter = myNeighbors.iterator();
		while (myNeighborsIter.hasNext()) {
		    CyNode  tempNode = myNeighborsIter.next();
		    if (!nodesToKeep.containsKey(tempNode.getSUID())) {
			nodesToKeep.put(tempNode.getSUID(), node);
		    }
		    }*/
	    }
	    else {
		System.out.println("Didn't find node for " + myInt);
	    }
	}
	int hide=0;
	int keep=0;
	for (View<CyNode> myNode : netView.getNodeViews()) {
	    if (!nodesToKeep.containsKey(myNode.getModel().getSUID())) {
		hide++;
		myNode.setVisualProperty(BasicVisualLexicon.NODE_VISIBLE, false);
	    }
	    else {
		myNode.setVisualProperty(BasicVisualLexicon.NODE_VISIBLE, true);
		keep++;
	    }
	}

	for (View<CyEdge> myEdge : netView.getEdgeViews()) {
	    if (nodesToKeep.containsKey(myEdge.getModel().getSource().getSUID()) &&
		nodesToKeep.containsKey(myEdge.getModel().getTarget().getSUID())) {

		// This line colorizes an edge, which will be useful for future features
		//myEdge.setVisualProperty(BasicVisualLexicon.EDGE_STROKE_UNSELECTED_PAINT, new Color(0,0,255));

		myEdge.setVisualProperty(BasicVisualLexicon.EDGE_VISIBLE, true);
	    }
	    else {
		myEdge.setVisualProperty(BasicVisualLexicon.EDGE_VISIBLE, false);
	    }
	}

	System.out.println("KEEP " + keep + " and HIDE " + hide);
	DisplayFullEmailDialog(email);
	revalidate();
	repaint();
	netView.setVisualProperty(BasicVisualLexicon.NETWORK_CENTER_X_LOCATION, nodeX);
	netView.setVisualProperty(BasicVisualLexicon.NETWORK_CENTER_Y_LOCATION, nodeY);
	netView.updateView();
    }

    public DoubleMatrix readFile(String filename, int rows, int isVector) {
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
	} finally {
	    try {
		if (br != null)br.close();
	    } catch (IOException ex) {
		ex.printStackTrace();
	    }
	}
	return tempMatrix;
    }
}
