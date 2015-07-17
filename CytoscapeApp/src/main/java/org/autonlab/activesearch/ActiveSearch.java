package org.autonlab.activesearch;

import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import org.cytoscape.app.swing.AbstractCySwingApp;
import org.cytoscape.app.swing.CySwingAppAdapter;
import org.cytoscape.model.CyEdge;
import org.cytoscape.model.CyNetwork;
import org.cytoscape.model.CyNetworkManager;
import org.cytoscape.model.events.NetworkAddedListener;
import org.cytoscape.property.CyProperty;
import org.cytoscape.property.SimpleCyProperty;
import org.cytoscape.service.util.CyServiceRegistrar;
import org.cytoscape.session.CySession;
import org.cytoscape.session.CySessionManager;
import org.cytoscape.task.NetworkViewTaskFactory;
import org.cytoscape.task.EdgeViewTaskFactory;
import org.cytoscape.task.NodeViewTaskFactory;
import static org.cytoscape.work.ServiceProperties.MENU_GRAVITY;
import static org.cytoscape.work.ServiceProperties.PREFERRED_ACTION;
import static org.cytoscape.work.ServiceProperties.PREFERRED_MENU;
import static org.cytoscape.work.ServiceProperties.TITLE;
import org.autonlab.activesearch.tasks.ShowFullEmailEdgeViewTaskFactory;
import org.autonlab.activesearch.tasks.ShowFullEmailNetworkViewTaskFactory;
import org.autonlab.activesearch.tasks.UpdateNodeNameViewTaskFactory;
import org.autonlab.activesearch.tasks.ShowAdjacentNodeViewTaskFactory;
import org.autonlab.activesearch.tasks.FilterMenuTaskFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.autonlab.activesearch.ActiveSearchConstants;
import org.autonlab.activesearch.ui.FullEmailDialog;

import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.application.CyApplicationManager;

public class ActiveSearch extends AbstractCySwingApp {

    public ActiveSearch(CySwingAppAdapter adapter)
    {
	super(adapter);	
	System.out.println("RUNNING HERE");

	final CyServiceRegistrar registrar = adapter.getCyServiceRegistrar();
	System.out.println("aaaa");
	Properties prop_show_email = new Properties();
	prop_show_email.setProperty(PREFERRED_MENU, "ActiveSearch[25]");
	prop_show_email.setProperty(MENU_GRAVITY, "6.0f");
	prop_show_email.setProperty(TITLE, "Show Emails Start With Random Seed");
	registrar.registerService(new ShowFullEmailNetworkViewTaskFactory(adapter), NetworkViewTaskFactory.class, prop_show_email);

	Properties prop_show_pos_email = new Properties();
	prop_show_email.setProperty(PREFERRED_MENU, "ActiveSearch[25]");
	prop_show_email.setProperty(MENU_GRAVITY, "7.0f");
	prop_show_email.setProperty(TITLE, "Show Positive Emails To Select Starting Seed");
	registrar.registerService(new ShowFullEmailNetworkViewTaskFactory(adapter, ActiveSearchConstants.MODE_SHOW_SELECT_SEED), NetworkViewTaskFactory.class, prop_show_email);

	Properties prop_show_email_prev = new Properties();
	prop_show_email_prev.setProperty(PREFERRED_MENU, "ActiveSearch[25]");
	prop_show_email_prev.setProperty(MENU_GRAVITY, "8.0f");
	prop_show_email_prev.setProperty(TITLE, "Show Emails Start With Previous Starting Seed");
	registrar.registerService(new ShowFullEmailNetworkViewTaskFactory(adapter, ActiveSearchConstants.MODE_SHOW_LAST_SEED), NetworkViewTaskFactory.class, prop_show_email_prev);

	Properties prop_show_email_filter = new Properties();
	prop_show_email_filter.setProperty(PREFERRED_MENU, "ActiveSearch[25]");
	prop_show_email_filter.setProperty(MENU_GRAVITY, "9.0f");
	prop_show_email_filter.setProperty(TITLE, "Show Emails Using Keyword Filter To Select Starting Seed");
	registrar.registerService(new FilterMenuTaskFactory(adapter), NetworkViewTaskFactory.class, prop_show_email_filter);

	Properties prop_show_email_edge = new Properties();
	prop_show_email_edge.setProperty(PREFERRED_MENU, "Active Search Edge Options[50]");
	prop_show_email_edge.setProperty(MENU_GRAVITY, "6.0f");
	prop_show_email_edge.setProperty(TITLE, "Show All Emails And Start New Search");
	registrar.registerService(new ShowFullEmailEdgeViewTaskFactory(adapter), EdgeViewTaskFactory.class, prop_show_email_edge);

	Properties show_adjacent_nodes = new Properties();
	show_adjacent_nodes.setProperty(PREFERRED_MENU, "Active Search Node Options[50]");
	show_adjacent_nodes.setProperty(MENU_GRAVITY, "7.0f");
	show_adjacent_nodes.setProperty(TITLE, "Show Only Adjacent Users");
	registrar.registerService(new ShowAdjacentNodeViewTaskFactory(adapter), NodeViewTaskFactory.class, show_adjacent_nodes);

	Properties update_node_names = new Properties();
	update_node_names.setProperty(PREFERRED_MENU, "Active Search Node Options[50]");
	update_node_names.setProperty(MENU_GRAVITY, "6.0f");
	update_node_names.setProperty(TITLE, "Replace User IDs with Names");
	registrar.registerService(new UpdateNodeNameViewTaskFactory(adapter), NodeViewTaskFactory.class, update_node_names);


	/*
	Properties filter_menu = new Properties();
	filter_menu.setProperty("preferredMenu", "Apps");
	filter_menu.setProperty("menuGravity", "6.0f");
	filter_menu.setProperty("title", "Filter Nodes");
	registrar.registerService(new FilterMenuTaskFactory(adapter), NetworkViewTaskFactory.class, filter_menu);

	FullEmailDialog dialog = new FullEmailDialog(networkView, null, adapter, xmappedMatrix, similaritySumMatrix, labelsMatrix);
	*/
    }
}
