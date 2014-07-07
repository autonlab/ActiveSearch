package org.autonlab.activesearch.ui;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.FlowLayout;

import javax.swing.BoxLayout;
import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JDialog;
import javax.swing.JTextField;

import java.util.LinkedList;
import java.util.TreeMap;
import java.util.Iterator;
import java.util.Collection;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.*;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.autonlab.activesearch.ActiveSearchConstants;
import org.autonlab.activesearch.DataConnectionRest;
import org.cytoscape.model.CyEdge.Type;
import org.cytoscape.application.CyApplicationManager;

import org.jblas.DoubleMatrix;
import org.cytoscape.view.presentation.property.BasicVisualLexicon;

public class FilterMenuDialog extends JDialog implements ActionListener {

    // JDialog requires this
    private static final long serialVersionUID = 1L;

    View<CyEdge> edgeView = null;
    CyNetworkView netView;
    DataConnectionRest dataConnectionHandle;

  
    JPanel panelFilter;
    JButton buttonFilterBySubjectText;
    JButton buttonFilterByEmailText;
    JTextField textSearchBox;

    GroupLayout layout;
    CySwingAppAdapter adapter;
    GroupLayout.SequentialGroup sGroup;

    public FilterMenuDialog(CyNetworkView myNetView, CySwingAppAdapter myAdapter)
    {
	super();

	this.netView = myNetView;
	adapter = myAdapter;

	dataConnectionHandle = new DataConnectionRest();

	/* the top panel where the email subjects are displayed */
	panelFilter = new JPanel();
	panelFilter.setOpaque(true);
 	panelFilter.setLayout(new BoxLayout(panelFilter, BoxLayout.Y_AXIS));

	textSearchBox = new JTextField(8);
	textSearchBox.setLocation(0,0);
	textSearchBox.setSize(100, 30);

	buttonFilterByEmailText = new JButton("Filter by email body");
	buttonFilterByEmailText.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent event){
		    LinkedList<String> emailList = dataConnectionHandle.getEmailsByKeyword(textSearchBox.getText());
		    FullEmailDialog emailDialog = new FullEmailDialog(netView, null, adapter, ActiveSearchConstants.MODE_DO_NOTHING_USER_WILL_PICK_SEED);
		    emailDialog.DisplayFullSubjectDialog(emailList);
		}
	    });

	buttonFilterBySubjectText = new JButton("Filter by subject");
	buttonFilterBySubjectText.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent event){
		    LinkedList<String> emailList = dataConnectionHandle.getEmailsByKeywordSubject(textSearchBox.getText());
		    FullEmailDialog emailDialog = new FullEmailDialog(netView, null, adapter, ActiveSearchConstants.MODE_DO_NOTHING_USER_WILL_PICK_SEED);
		    emailDialog.DisplayFullSubjectDialog(emailList);
		}
	    });

	panelFilter.add(textSearchBox);
	panelFilter.add(buttonFilterBySubjectText);
	panelFilter.add(buttonFilterByEmailText);

	layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setAutoCreateGaps(false);
        layout.setAutoCreateContainerGaps(true);
 	
	GroupLayout.ParallelGroup pGroup = layout.createParallelGroup(GroupLayout.Alignment.LEADING);
	pGroup.addComponent(panelFilter);
	layout.setHorizontalGroup(layout.createSequentialGroup().addGroup(pGroup));

	sGroup = layout.createSequentialGroup();
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(panelFilter));
        layout.setVerticalGroup(sGroup);

	pack();
	setTitle("Keyword Search");
	setAlwaysOnTop(true);
	setResizable(true);
	setLocationRelativeTo(null);
	setVisible(true);
	setLocation(850, 650);
    }

    public void actionPerformed(ActionEvent e) {
	//
    }


}
