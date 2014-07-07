package org.autonlab.activesearch.ui;

import java.awt.Dimension;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import javax.swing.BoxLayout;
import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import java.awt.Component;

import java.util.LinkedList;
import java.util.Iterator;

import org.autonlab.activesearch.DataConnectionRest;
import org.autonlab.activesearch.ui.DisplayOneEmail;
import org.autonlab.activesearch.ActiveSearchConstants;

public class StatisticsDialog extends JDialog implements ActionListener {

    // JDialog requires this
    private static final long serialVersionUID = 1L;

    JPanel panelBody;
    JScrollPane panelBodyScrollpane;

    JLabel labelEmailCount;
    String emailCountPrefix;
    int emailCount;

    JLabel labelPositiveCount;
    String positiveCountPrefix;
    int positiveCount;

    JLabel labelNegativeCount;
    String negativeCountPrefix;
    int negativeCount;

    JLabel labelSeed;
    String seedPrefix;

    LinkedList<String> emailHistory;

    DataConnectionRest dataConnectionHandle;

    GroupLayout layout;

    DisplayOneEmail emailWindow;

    FullEmailDialog parentDialog;

    public StatisticsDialog(DataConnectionRest handle, FullEmailDialog myParent)
    {
	parentDialog = myParent;
	StatisticsDialogCommon(handle, 0);
    }

    public StatisticsDialog(DataConnectionRest handle, FullEmailDialog myParent, int option)
    {
	parentDialog = myParent;
	StatisticsDialogCommon(handle, option);
    }

    private void StatisticsDialogCommon(DataConnectionRest handle, int option)
    {
	emailCountPrefix = "Emails Seen: ";
	positiveCountPrefix = "Interesting Emails Seen: ";
	negativeCountPrefix = "Boring Emails Seen: ";
	seedPrefix = "Starting Seed Email ID: ";

	dataConnectionHandle = handle;

	emailCount = 0;
	positiveCount = 0;
	negativeCount = 0;

	panelBody = new JPanel();
	panelBody.setOpaque(true);
	panelBody.setLayout(new BoxLayout(panelBody, BoxLayout.Y_AXIS));

	panelBodyScrollpane = new JScrollPane(panelBody, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
					      JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	panelBodyScrollpane.setPreferredSize(new Dimension(640, 480));

	labelEmailCount = new JLabel();
	labelEmailCount.setAlignmentX(Component.LEFT_ALIGNMENT);
	labelEmailCount.setVisible(true);
	labelEmailCount.setText(emailCountPrefix + Integer.toString(emailCount));
	panelBody.add(labelEmailCount);

	labelPositiveCount = new JLabel();
	labelPositiveCount.setAlignmentX(Component.LEFT_ALIGNMENT);
	labelPositiveCount.setVisible(true);
	labelPositiveCount.setText(positiveCountPrefix + Integer.toString(positiveCount));
	panelBody.add(labelPositiveCount);

	labelNegativeCount = new JLabel();
	labelNegativeCount.setAlignmentX(Component.LEFT_ALIGNMENT);
	labelNegativeCount.setVisible(true);
	labelNegativeCount.setText(negativeCountPrefix + Integer.toString(negativeCount));
	panelBody.add(labelNegativeCount);

	labelSeed = new JLabel();
	labelSeed.setAlignmentX(Component.LEFT_ALIGNMENT);
	labelSeed.setVisible(true);
	labelSeed.setText(seedPrefix + Integer.toString(dataConnectionHandle.getStartPoint()));
	panelBody.add(labelSeed);

	if (option == ActiveSearchConstants.HIDE_SEED_FIELD) {
	    labelSeed.setVisible(false);
	}

	panelBody.setVisible(true);

	layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setAutoCreateGaps(false);
        layout.setAutoCreateContainerGaps(true);

	GroupLayout.ParallelGroup pGroup = layout.createParallelGroup(GroupLayout.Alignment.LEADING);
	pGroup.addComponent(panelBodyScrollpane);
	layout.setHorizontalGroup(layout.createSequentialGroup().addGroup(pGroup));

	GroupLayout.SequentialGroup sGroup = layout.createSequentialGroup();
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(panelBodyScrollpane));
        layout.setVerticalGroup(sGroup);

	pack();
	setAlwaysOnTop(false);
	setResizable(true);
	setLocationRelativeTo(null);
	setVisible(true);
	setLocation(850, 0);

	emailHistory = new LinkedList<String>();
    }

    public void addEmail(int email, int vote, int isRevote, int oldVote) {
	if (isRevote == 0) {
	    emailCount++;
	    if (vote == 0) {
		negativeCount++;
	    }
	    else {
		positiveCount++;
	    }
	}
	else {
	    if (vote == 0 && oldVote == 1) {
		positiveCount--;
		negativeCount++;
	    }
	    else if (vote == 1 && oldVote == 0) {
		positiveCount++;
		negativeCount--;
	    }
	}

	emailHistory.add(email + " : " + ((isRevote == 1) ? "(revote) " : "") +
			 ((vote == 1) ? "Interesting" : "Boring") + " : " +
			 dataConnectionHandle.getEmailSubjectFromMessageID(email));

	labelEmailCount.setText(emailCountPrefix + Integer.toString(emailCount));
	labelPositiveCount.setText(positiveCountPrefix + Integer.toString(positiveCount));
	labelNegativeCount.setText(negativeCountPrefix + Integer.toString(negativeCount));
	labelSeed.setText(seedPrefix + Integer.toString(dataConnectionHandle.getStartPoint()));
	labelSeed.setVisible(true); //possiblly hidden in constructor depending on options

	panelBody.removeAll();
	panelBody.add(labelEmailCount);
	panelBody.add(labelPositiveCount);
	panelBody.add(labelNegativeCount);
	panelBody.add(labelSeed);
	
	for (String b : emailHistory) {
	    JButton temp = new JButton();
	    temp.setText(b);
	    temp.addActionListener(this);
	    panelBody.add(temp);
	}

	revalidate();
	repaint();
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

	// this makes a new window for each button press
	emailWindow = new DisplayOneEmail(dataConnectionHandle, parentDialog);
	emailWindow.showEmail(email);
    }

    public void showSeed() {
	labelSeed.setText(seedPrefix + Integer.toString(dataConnectionHandle.getStartPoint()));
	labelSeed.setVisible(true);
    }
}