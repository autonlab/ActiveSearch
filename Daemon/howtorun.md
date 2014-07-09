# How to Run the ActiveSearch Daemon

## Make sure these are installed:
 * mysql
 * Oracle Java 7 and jdk
 * Perl
 * Apache Maven
 * libgfortran (Without this the java jblas library will fail in a not-obvious manner, so make sure not to skip this step! This is probably available via yum or apt repository.) On Ubuntu 14.04 the package is called "libgfortran3"

## Set Up the Database and Perform the One-Time Precomputation
 1. Set up a mysql database. Create a database called "scottwalker". For simplicity we use a username of "root" and a blank password.
 2. Switch to the ActiveSearch/Database directory. Import the dataset into the database: perl import_database.pl -file=scottwalker.tsv -database=scottwalker
      - edit $DATABASE_USERNAME or $DATABASE_PASSWORD in import_database.pl as necessary
      - this dataset will require 30GB of ram to do the precomputation. Optionally, use scottwalker_5000.tsv which only contains the first 5000 emails and
             only requires about 6GB of RAM
 3. Switch to the ActiveSearch/Daemon directory. Modify the database info if necessary in src/main/java/org/autonlab/activesearch/daemon/ActiveSearchConstants.java
 4. Set your shell's environment so that Maven will have enough memory to do the precomputation: export MAVEN_OPTS=-Xmx30g (or 6g depending on step 3)
 5. Start the REST daemon:
      `> mvn clean; mvn tomcat:run`
 6. Begin the prcomputation by pointing your browser to http://localhost:8080/ActiveSearchDaemon/rest/eigenmap/<#> (replace <#> with the number of threads to use).
     - On an Intel i7 4770K CPU, this takes a little over half an hour. Your browser might timeout waiting for it. Either lengthen the timeout or just wait until the daemon process stops taking up CPU cycles. This will create two files in the directory from which you ran your daemon: similarity_rowsum.out and X_matrix.out
 7. Kill the REST daemon (the precomputation doesn't free up its memory but I haven't figured out why)

## Start the REST daemon for general use
 1. mvn clean; mvn tomcat:run
 2. Initialize the Active Search system with an initial seed email:  http://localhost:8080/ActiveSearchDaemon/rest/firstemail/< email ID >
     - the email ID is in the database as messages.messageid (which is also the zero-indexed row number of the original tsv file)
     - Then call http://localhost:8080/ActiveSearchDaemon/rest/getNextEmail to get the first email recommendation
 3. For the email returned in step 2., tell the system if it is interesting (http://localhost:8080/ActiveSearchDaemon/rest/emailinteresting) or not
      (http://localhost:8080/ActiveSearchDaemon/rest/emailboring). Each of these calls will return the next email ID that it recommends


For an optional GUI that can talk to this REST daemon, see the [Cytoscape App](https://github.com/AutonlabCMU/ActiveSearch/blob/master/CytoscapeApp/howtorun.md)
