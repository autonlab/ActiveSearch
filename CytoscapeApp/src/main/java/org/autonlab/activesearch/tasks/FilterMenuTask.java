package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNode;
import org.cytoscape.model.CyIdentifiable;
import org.cytoscape.model.CyNetwork;
import org.cytoscape.task.AbstractNetworkViewTask;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskMonitor;
import org.cytoscape.view.presentation.property.BasicVisualLexicon;

import org.autonlab.activesearch.ui.FilterMenuDialog;

public class FilterMenuTask extends AbstractNetworkViewTask{

    public static CySwingAppAdapter adapter;
    public static CyNetworkView netView;

    public FilterMenuTask(CyNetworkView netView, CySwingAppAdapter myAdapter)
    {
	super(netView);	
	FilterMenuTask.netView = netView;
	adapter = myAdapter;
    }
	
    public void run(TaskMonitor tm) throws Exception {
	FilterMenu(netView);
    }

    public static void FilterMenu(CyNetworkView myNetView) {
	new FilterMenuDialog(netView, adapter);
    }
}

