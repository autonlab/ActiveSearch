package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNode;
import org.cytoscape.task.AbstractEdgeViewTask;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskMonitor;
import org.autonlab.activesearch.ActiveSearchConstants;
import org.autonlab.activesearch.ui.FullEmailDialog;

public class ShowFullEmailEdgeViewTask extends AbstractEdgeViewTask{

    static CyNetworkView netView;
    static View<CyEdge> edgeView = null;
    static CySwingAppAdapter adapter;
    
    public ShowFullEmailEdgeViewTask(View<CyEdge> edgeView, CyNetworkView netView, CySwingAppAdapter myAdapter, int myMode)
    {
	super(edgeView,netView);
	ShowFullEmailEdgeViewTask.netView = netView;
	ShowFullEmailEdgeViewTask.edgeView = edgeView;
	adapter = myAdapter;
    }

    public void run(TaskMonitor tm) throws Exception {
	new FullEmailDialog(netView, edgeView, adapter, ActiveSearchConstants.MODE_SHOW_EDGE_EMAILS);
    }
}

