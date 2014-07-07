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

public class LoadingDialog extends JDialog implements ActionListener {

    // JDialog requires this
    private static final long serialVersionUID = 1L;

    JLabel labelLoading;
    GroupLayout layout;

    JButton buttonClose;


    public LoadingDialog(String message)
    {

	buttonClose = new JButton();
	buttonClose.setText("Close This Window");
	buttonClose.addActionListener(this);

	// disable this because the cytoscape task loading causes the window to freeze up
	// so the button is unresponsive anyway
	// The caller should just call closeWindow()
	buttonClose.setVisible(false); 

	labelLoading = new JLabel();
	labelLoading.setText(message);
	labelLoading.setVisible(true);

	layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setAutoCreateGaps(false);
        layout.setAutoCreateContainerGaps(true);

	GroupLayout.ParallelGroup pGroup = layout.createParallelGroup(GroupLayout.Alignment.LEADING);
	pGroup.addComponent(labelLoading);
	pGroup.addComponent(buttonClose);
	layout.setHorizontalGroup(layout.createSequentialGroup().addGroup(pGroup));

	GroupLayout.SequentialGroup sGroup = layout.createSequentialGroup();
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(labelLoading));
	sGroup.addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
			.addComponent(buttonClose));
        layout.setVerticalGroup(sGroup);

	pack();
	setAlwaysOnTop(true);
	setResizable(true);
	setLocationRelativeTo(null);
	setVisible(true);
	setLocation(600, 400);
	revalidate();
	repaint();
    }

    public void closeWindow() {
	dispose();
    }

    public void actionPerformed(ActionEvent e) {
	dispose();
    }
}