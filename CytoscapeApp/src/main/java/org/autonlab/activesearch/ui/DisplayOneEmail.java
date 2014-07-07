package org.autonlab.activesearch.ui;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BoxLayout;
import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import java.awt.Component;

import java.util.LinkedList;
import java.util.Iterator;

import org.autonlab.activesearch.tasks.ShowFullEmailEdgeViewTask;
import org.autonlab.activesearch.DataConnectionRest;

public class DisplayOneEmail extends JDialog  implements ActionListener {

    // JDialog requires this
    private static final long serialVersionUID = 1L;

    DataConnectionRest dataConnectionHandle;

    JTextArea areaEmailContents;
    JScrollPane panelEmailBodyScrollpane;
    JPanel panelEmailBody;
    JLabel labelSender;
    JLabel labelReceivers;
    JLabel labelSubject;
    JLabel labelTime;
    JLabel labelASLabel;

    JButton buttonReVote;

    GroupLayout layout;

    FullEmailDialog parentDialog;

    int currentEmail;

    public DisplayOneEmail(DataConnectionRest handle, FullEmailDialog myParent)
    {
	dataConnectionHandle = handle;

	parentDialog = myParent;

	panelEmailBody = new JPanel();
	panelEmailBody.setOpaque(true);
	panelEmailBody.setLayout(new BoxLayout(panelEmailBody, BoxLayout.Y_AXIS));
	//panelEmailBody.setLayout(new FlowLayout(FlowLayout.LEFT));

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

	buttonReVote = new JButton("Show this email in Recommendations Window for re-vote");
	buttonReVote.addActionListener(this);
	buttonReVote.setVisible(true);

	layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setAutoCreateGaps(false);
        layout.setAutoCreateContainerGaps(true);
 	
	GroupLayout.ParallelGroup pGroup = layout.createParallelGroup(GroupLayout.Alignment.LEADING);
	pGroup.addComponent(panelEmailBodyScrollpane);
	pGroup.addComponent(buttonReVote);
	layout.setHorizontalGroup(layout.createSequentialGroup().addGroup(pGroup));

	GroupLayout.SequentialGroup sGroup = layout.createSequentialGroup();
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(panelEmailBodyScrollpane));
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(buttonReVote));
        layout.setVerticalGroup(sGroup);

	pack();
	setAlwaysOnTop(false);
	setResizable(true);
	setLocationRelativeTo(null);
	setVisible(true);
	setLocation(1000, 300);
    }

    /**
     * This is a separate function from the constructor because some day we might want to
     * replace the email within the same window rather than open a new one
     */
    public void showEmail(int email) {
	currentEmail = email;

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
	labelASLabel.setText("label: " + dataConnectionHandle.getLabel(email));

	labelSender.setVisible(true);
	labelReceivers.setVisible(true);
	labelTime.setVisible(true);
	labelSubject.setVisible(true);
	labelASLabel.setVisible(true);
	panelEmailBody.setVisible(true);
	panelEmailBodyScrollpane.setVisible(true);
	setTitle("Viewing Previously Seen Email ID " + email);

	areaEmailContents.setText(dataConnectionHandle.getEmailBodyFromMessageID(email));
	areaEmailContents.setCaretPosition(0);

	revalidate();
	repaint();
    }

    public void actionPerformed(ActionEvent e) {
	parentDialog.setNextEmailIsReVote();
	parentDialog.hideDistantEmails(currentEmail);
	dispose();
    }
}
