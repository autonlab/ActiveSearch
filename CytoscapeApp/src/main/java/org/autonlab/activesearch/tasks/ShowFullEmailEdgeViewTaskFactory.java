package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNode;
import org.cytoscape.model.CyRow;
import org.cytoscape.task.AbstractEdgeViewTaskFactory;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskIterator;

public class ShowFullEmailEdgeViewTaskFactory extends AbstractEdgeViewTaskFactory {

    CySwingAppAdapter adapter;
    int mode;

    /*
     * Default constructor. This enters the main loop using a randomly selected positively labeled initial email
     */
    public ShowFullEmailEdgeViewTaskFactory(CySwingAppAdapter adapter){
	this.adapter = adapter;
	mode = 0;
    }

    /*
     * @param myMode 0 = default, 1 = show positively labeled emails then user enters main loop when they find one they like
     */
    public ShowFullEmailEdgeViewTaskFactory(CySwingAppAdapter adapter, int myMode){
	this.adapter = adapter;
	mode = myMode;
    }
	
    public boolean isReady(View<CyEdge> edgeView, CyNetworkView networkView) {
	return true;
    }

    public TaskIterator createTaskIterator(View<CyEdge> edgeView, CyNetworkView networkView) {
	return new TaskIterator(new ShowFullEmailEdgeViewTask(edgeView, networkView, adapter, mode));
    }
}
