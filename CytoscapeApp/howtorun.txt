Running Cytoscape:
 1. Add the Cytoscape API jar file to your maven repository. For this example we'll use 3.1.0:
     Download the file from http://www.cytoscape.org/documentation_developers.html
        (At the very bottom of the page is a link for "API Jar for 3.1.0")
     mvn install:install-file -Dfile=cytoscape-swing-app-api-3.1.0.jar -DgroupId=org.cytoscape -DartifactId=cytoscape3-api -Dversion=3.1.0 -Dpackaging=jar
 2. cd to the ActiveSearch/CytoscapeApp directory and modify as necessary src/main/java/org/autonlab/activesearch/ActiveSearchConstants.java 
 4. mvn clean; mvn package (this creates h/ActiveSearch/CytoscapeApp/target/ActiveSearchDaemon-1.0.jar)
 5. Get these jar files from your maven repository and copy them to <Cytoscape install dir>/framework/lib/
     jblas
     jersey-client
     jersey-core
     - To locate the appropriate jar files, go into ~/.m2/repository and do a "find" for the above names
 6. Start Cytoscape with ./Cytoscape -s Database/scottwalker_cytoscape.cys (or scottwalker_5000_cytoscape.cys depending on your database). Each node is a person and each edge represents the emails between them
 7. Within Cytoscape, install CytoscapeApp/target/ActiveSearchDaemon-1.0.jar as an app
At this point there are a few ways to seed the tool with initial email. Try right-clicking on a node or an edge or try the ActiveSearch menu in the toolbar. We'll give one example here:
 8. Under the ActiveSearch menu, select "Show Emails Using Keyword Filter To Select Starting Seed". Search the subjects for "odonnel". 
 9. Select one of the emails from the resulting list to view it, then click "Start Making Recommendations"
 10. Click "Interesting" or "Boring" to see the next recommendation. The slider bar affects how far away from the current email the algorithm will wander when looking for related emails. 
