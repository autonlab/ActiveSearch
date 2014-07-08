# ActiveSearch Overview

ActiveSearch takes a collection of emails (or any dataset where a
similarity can be generated between elements) and recommends related
messages based on user feedback. The user provides an initial seed
email then enters into a cycle where ActiveSearch provides a similar
email and the user reports whether or not the email was interesting.

ActiveSearch is useful for anyone navigating a large set of emails and looking for related messages on a specific topic. As it considers the similarities between emails as well as user feedback, it is an improvement in accuracy, time, and effort over basic text search or a brute force search.

## Skill Sets:

Required:
 * Linux (the tool is written using cross-platform tools but is only tested on Linux)
 * Mysql (the emails are stored in a database)
 * REST (used to interact with the tool)
  
Optional:
 * Cytoscape (an optional UI we provide that speaks the tool's REST API)
 * Perl (if importing different data into the database)
 * Java (for changing the similarity comparison algorithm)
