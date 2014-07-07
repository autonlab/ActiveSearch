package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyNode;
import org.cytoscape.task.AbstractNodeViewTaskFactory;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskIterator;

import org.jblas.DoubleMatrix;

import java.net.URI;

public class UpdateNodeNameViewTaskFactory extends AbstractNodeViewTaskFactory {

    CySwingAppAdapter adapter;

    public UpdateNodeNameViewTaskFactory(CySwingAppAdapter adapter){
	this.adapter = adapter;
    }
	
    public boolean isReady(View<CyNode> nodeView, CyNetworkView networkView) {
	return true;
    }

    public TaskIterator createTaskIterator(View<CyNode> nodeView, CyNetworkView networkView) {
	return new TaskIterator(new UpdateNodeNameViewTask(nodeView, networkView, adapter));
    }
}
