package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNode;
import org.cytoscape.model.CyRow;
import org.cytoscape.task.AbstractNetworkViewTaskFactory;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskIterator;

public class FilterMenuTaskFactory extends AbstractNetworkViewTaskFactory {

    CySwingAppAdapter adapter;

    public FilterMenuTaskFactory(CySwingAppAdapter adapter){
	this.adapter = adapter;
    }
	
    public boolean isReady(CyNetworkView networkView) {
	return true;
    }

    public TaskIterator createTaskIterator(CyNetworkView networkView) {
	return new TaskIterator(new FilterMenuTask(networkView, adapter));
    }
}
