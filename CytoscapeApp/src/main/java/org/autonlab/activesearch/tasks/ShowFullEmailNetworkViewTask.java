package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNode;
import org.cytoscape.task.AbstractNetworkViewTask;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskMonitor;
import org.autonlab.activesearch.ui.FullEmailDialog;

public class ShowFullEmailNetworkViewTask extends AbstractNetworkViewTask{

    static CyNetworkView netView;
    static View<CyEdge> edgeView = null;
    static CySwingAppAdapter adapter;
    
    int mode;

    public ShowFullEmailNetworkViewTask(View<CyEdge> edgeView, CyNetworkView netView, CySwingAppAdapter myAdapter, int myMode)
    {
	super(netView);	
	ShowFullEmailNetworkViewTask.netView = netView;
	ShowFullEmailNetworkViewTask.edgeView = edgeView;
	adapter = myAdapter;
	mode = myMode;
    }
	
    public void run(TaskMonitor tm) throws Exception {
	new FullEmailDialog(netView, edgeView, adapter, mode);
    }
}

