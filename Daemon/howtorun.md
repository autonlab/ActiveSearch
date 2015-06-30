# How to Run the ActiveSearch Daemon

## (6/30/2015: these instructions were out of date so I have removed them until I write something new)


For an optional GUI that can talk to this REST daemon, see the [Cytoscape App](https://github.com/AutonlabCMU/ActiveSearch/blob/master/CytoscapeApp/howtorun.md)

## Customizing the import_database.pl tool 
If you have a tsv file with your data and it is formatted in the same way as the scott walker emails, import_database.pl will work. The format is:

zero-indexed column  value
3                    datetime
5                    sender
7                    csv of 'to' recipients
8                    csv of 'cc' recipients
9                    csv of 'bcc' recipients
14                   subject
15                   body

If your dataset is already in a tsv, the column numbers can be passed to import_database.pl to replace the default column numbers listed above. 
Run import_database.pl with no arguments for more information
