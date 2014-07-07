package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNode;
import org.cytoscape.model.CyIdentifiable;
import org.cytoscape.model.CyNetwork;
import org.cytoscape.task.AbstractNodeViewTask;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskMonitor;
import org.cytoscape.view.presentation.property.BasicVisualLexicon;

import java.net.URI;

import com.sun.jersey.api.client.Client;
import com.sun.jersey.api.client.ClientResponse;
import com.sun.jersey.api.client.WebResource;

import org.autonlab.activesearch.ActiveSearchConstants;

import org.autonlab.activesearch.DataConnectionRest;

public class UpdateNodeNameViewTask extends AbstractNodeViewTask{
    public static int done = 0;
    public static CyNetworkView netView;
    public static View<CyNode> nodeView = null;
    CySwingAppAdapter adapter;

    public UpdateNodeNameViewTask(View<CyNode> nodeView, CyNetworkView netView, CySwingAppAdapter adapter)
    {
	super(nodeView,netView);	
	UpdateNodeNameViewTask.netView = netView;
	UpdateNodeNameViewTask.nodeView = nodeView;
	this.adapter = adapter;
    }
	
    public void run(TaskMonitor tm) throws Exception {
	// This is called by the user from the UI so reset "done" to force an update
	done = 0;
	UpdateNodeNames(netView);
    }

    public static void UpdateNodeNames(CyNetworkView myNetView) {
	if (done == 1) {
	    return;
	}

	CyNetwork myNetwork = myNetView.getModel();
	DataConnectionRest dataConnection = new DataConnectionRest();

	for (View<CyNode> myNode : myNetView.getNodeViews()) {
	    int nodeID = Integer.parseInt(myNetwork.getRow(myNode.getModel()).get(CyNetwork.NAME, String.class).replaceAll("\"", ""));
	    String userName = dataConnection.getUserNameFromID(nodeID);
	    myNode.setVisualProperty(BasicVisualLexicon.NODE_LABEL, userName);
	    myNode.setVisualProperty(BasicVisualLexicon.NODE_WIDTH, userName.length() * 7.5);
	}
	myNetView.updateView();
	done = 1;
    }
  
}

