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

import java.util.Iterator;
import java.util.TreeMap;
import java.util.List;

public class ShowAdjacentNodeViewTask extends AbstractNodeViewTask{
    public static CyNetworkView netView;
    public static View<CyNode> nodeView = null;
    CySwingAppAdapter adapter;

    public ShowAdjacentNodeViewTask(View<CyNode> nodeView, CyNetworkView netView, CySwingAppAdapter adapter)
    {
	super(nodeView,netView);	
	ShowAdjacentNodeViewTask.netView = netView;
	ShowAdjacentNodeViewTask.nodeView = nodeView;
	this.adapter = adapter;
    }
	
    public void run(TaskMonitor tm) throws Exception {
	ShowAdjacentNodes(netView);
    }

    public static void ShowAdjacentNodes(CyNetworkView myNetView) {
	TreeMap<Long, CyNode> nodesToKeep = new TreeMap<Long, CyNode>();

	CyNetwork myNetwork = myNetView.getModel();
	CyNode node = nodeView.getModel();

	// include the current node in the list of nodes to keep
	nodesToKeep.put(node.getSUID(), node);

	// get all neighbors of current node
	List<CyNode> myNeighbors = myNetwork.getNeighborList(node, CyEdge.Type.ANY);
	Iterator<CyNode> myNeighborsIter = myNeighbors.iterator();
	while (myNeighborsIter.hasNext()) {
	    CyNode  tempNode = myNeighborsIter.next();
	    if (!nodesToKeep.containsKey(tempNode.getSUID())) {
		nodesToKeep.put(tempNode.getSUID(), tempNode);
	    }
	}

	// for each node in the view, set it visible or not depending on nodesToKeep
	for (View<CyNode> myNode : myNetView.getNodeViews()) {
	    if (!nodesToKeep.containsKey(myNode.getModel().getSUID())) {
		myNode.setVisualProperty(BasicVisualLexicon.NODE_VISIBLE, false);
	    }
	    else {
		myNode.setVisualProperty(BasicVisualLexicon.NODE_VISIBLE, true);
	    }
	}

	// for each edge in the view, set it visible or not depending on whether the endpoints are in nodesToKeep
	for (View<CyEdge> myEdge : myNetView.getEdgeViews()) {
	    if (nodesToKeep.containsKey(myEdge.getModel().getSource().getSUID()) &&
		nodesToKeep.containsKey(myEdge.getModel().getTarget().getSUID())) {
		myEdge.setVisualProperty(BasicVisualLexicon.EDGE_VISIBLE, true);
	    }
	    else {
		myEdge.setVisualProperty(BasicVisualLexicon.EDGE_VISIBLE, false);
	    }
	}

	myNetView.updateView();
    }
}

