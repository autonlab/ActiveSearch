package org.autonlab.activesearch.tasks;

import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNode;
import org.cytoscape.model.CyRow;
import org.cytoscape.task.AbstractNetworkViewTaskFactory;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.TaskIterator;

public class ShowFullEmailNetworkViewTaskFactory extends AbstractNetworkViewTaskFactory {

    CySwingAppAdapter adapter;
    int mode;

    /*
     * Default constructor. This enters the main loop using a randomly selected positively labeled initial email
     */
    public ShowFullEmailNetworkViewTaskFactory(CySwingAppAdapter adapter){
	this.adapter = adapter;
	mode = 0;
    }

    /*
     * @param myMode 0 = default, 1 = show positively labeled emails then user enters main loop when they find one they like
     */
    public ShowFullEmailNetworkViewTaskFactory(CySwingAppAdapter adapter, int myMode){
	this.adapter = adapter;
	mode = myMode;
    }
	
    public boolean isReady(View<CyEdge> edgeView, CyNetworkView networkView) {
	return true;
    }

    public TaskIterator createTaskIterator(CyNetworkView networkView) {
	return new TaskIterator(new ShowFullEmailNetworkViewTask(null, networkView, adapter, mode));
    }
}
