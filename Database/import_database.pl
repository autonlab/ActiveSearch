#!/usr/bin/perl -w
use DBI;
use strict;
use Getopt::Long;
use Lingua::Stem::Snowball;

my $old_fh = select(STDOUT);
$| = 1;
select($old_fh);

sub insertRecipientFromArray($$);
sub getUserIDFromEmail($);
sub getWordID($);
sub setTFIDF();
sub removeTFIDFThreshold($);

# This script can be used to import a tab-separated value file where each row represents one email
# Fill in these constants indicating the zero-based index where the necessary fields reside
# For the Scott Walker dataset that we had, the fields looked like this:
# num	dir	category	datetime	importance	from	ip	to	cc	bcc	attachments	messageid	inreplyto	references	subject	body
my $DATETIME_INDEX = 3;
my $SENDER_INDEX = 5;
my $TO_LIST_INDEX = 7; #comma or semicolon separated list
my $CC_LIST_INDEX = 8; #comma or semicolon separated list
my $BCC_LIST_INDEX = 9; #comma or semicolon separated list
my $SUBJECT_INDEX = 14;
my $BODY_INDEX = 15;
my $DATABASE_USERNAME = "";
my $DATABASE_PASSWORD = "";

# remove the least frequently seen words so that there are approximately this many left
my $tfidf_wordlimit=2000;

# remove the least frequently seen users so that there are approximately this many left
my $user_limit=600;

sub printOpts() {
  print "  -file=<file>  The path to the tsv file containing the email information\n";
  print "  -database=<db> The database to use\n";
  print "  -database_username=<username> User name used to log into the database\n";
  print "  -database_password=<password> Password used to log into the database\n";
  print "  -COL_DATETIME=<#> The zero-indexed column in the tsv containing the message datetime stamp\n";
  print "  -COL_SENDER=<#> The zero-indexed column in the tsv containing the sender\n";
  print "  -COL_TO=<#> The zero-indexed column in the tsv containing the csv of 'to' recipients\n";
  print "  -COL_CC=<#> The zero-indexed column in the tsv containing the csv of 'cc' recipients\n";
  print "  -COL_BCC=<#> The zero-indexed column in the tsv containing the csv of 'bcc' recipients\n";
  print "  -COL_SUBJECT=<#> The zero-indexed column in the tsv containing the subject\n";
  print "  -COL_BODY=<#> The zero-indexed column in the tsv containing the body\n";
  print "  -wordlimit=<#> Keep the number of tfidf words below this. 0=skip\n";
  print "  -userlimit=<#> Keep the number of users below this. 0=skip\n";
  print "  -dotfidf Do tfidf calculation. About 5x slower than having daemon do it, but results are saved so it is only done once\n";
}

my $dotfidf = undef;

my $TSV_FILE_NAME = "scottwalker.tsv";
my $DATABASE_NAME = "";
if (!GetOptions ("file=s" => \$TSV_FILE_NAME,
		"database=s" => \$DATABASE_NAME,
		"database_username=s" => \$DATABASE_USERNAME,
		"database_password=s" => \$DATABASE_PASSWORD,
		"COL_DATETIME=i" => \$DATETIME_INDEX,
		"COL_SENDER=i" => \$SENDER_INDEX,
		"COL_TO=i" => \$TO_LIST_INDEX,
		"COL_CC=i" => \$CC_LIST_INDEX,
		"COL_BCC=i" => \$BCC_LIST_INDEX,
		"COL_SUBJECT=i" => \$SUBJECT_INDEX,
		"COL_BODY=i" => \$BODY_INDEX,
                "wordlimit=i" => \$tfidf_wordlimit,
                "userlimit=i" => \$user_limit,
                "dotfidf" => \$dotfidf)) {
    printOpts();
    die ("Error in command line arguments\n");
}

if ($DATABASE_NAME eq "" || $TSV_FILE_NAME eq "") {
    printOpts();
    die ("database name and file name must be set");
}
if ($DATABASE_USERNAME eq "") {
    printOpts();
    die ("database username must be set");
}

my $stemmer = Lingua::Stem::Snowball->new(
    lang     => 'es', 
    encoding => 'UTF-8',
    );
die $@ if $@;

my %skip_words = ();
$skip_words{'the'} = 1;
$skip_words{'be'} = 1;
$skip_words{'to'} = 1;
$skip_words{'of'} = 1;
$skip_words{'and'} = 1;
$skip_words{'a'} = 1;
$skip_words{'in'} = 1;
$skip_words{'that'} = 1;
$skip_words{'have'} = 1;
$skip_words{'i'} = 1;
$skip_words{'it'} = 1;
$skip_words{'for'} = 1;
$skip_words{'not'} = 1;
$skip_words{'on'} = 1;
$skip_words{'with'} = 1;
$skip_words{'he'} = 1;
$skip_words{'as'} = 1;
$skip_words{'you'} = 1;
$skip_words{'do'} = 1;
$skip_words{'at'} = 1;

$skip_words{'this'} = 1;
$skip_words{'but'} = 1;
$skip_words{'his'} = 1;
$skip_words{'by'} = 1;
$skip_words{'from'} = 1;
$skip_words{'they'} = 1;
$skip_words{'we'} = 1;
$skip_words{'say'} = 1;
$skip_words{'her'} = 1;
$skip_words{'she'} = 1;
$skip_words{'or'} = 1;
$skip_words{'an'} = 1;
$skip_words{'will'} = 1;
$skip_words{'my'} = 1;
$skip_words{'one'} = 1;
$skip_words{'all'} = 1;
$skip_words{'would'} = 1;
$skip_words{'there'} = 1;
$skip_words{'their'} = 1;
$skip_words{'what'} = 1;

my $wordIndex = 0;
my %word_id_cache = ();

my $dbh = DBI->connect("dbi:mysql:" . $DATABASE_NAME, $DATABASE_USERNAME, $DATABASE_PASSWORD,  {RaiseError => 1, PrintError => 1, mysql_enable_utf8 => 1})
    or die "Connection Error: $DBI::errstr\n";

$dbh->do("DROP TABLE IF EXISTS bodies");
$dbh->do("DROP TABLE IF EXISTS messages");
$dbh->do("DROP TABLE IF EXISTS people");
$dbh->do("DROP TABLE IF EXISTS recipients");
$dbh->do("DROP TABLE IF EXISTS tf_idf_dictionary");
$dbh->do("DROP TABLE IF EXISTS tf_idf_wordmap");

$dbh->do("CREATE TABLE bodies (messageid int(10), body text, PRIMARY KEY (messageid)) ENGINE=MyISAM");
$dbh->do("CREATE TABLE messages (messageid int(10), messagedt timestamp, senderid int(10), subject varchar(255), PRIMARY KEY (messageid), INDEX (senderid), INDEX(subject)) ENGINE=MyISAM");
$dbh->do("CREATE TABLE people (personid int(10), email varchar(255), name varchar(255), PRIMARY KEY (personid), INDEX (email), INDEX(name)) ENGINE=MyISAM");
$dbh->do("CREATE TABLE recipients (messageid int(10), personid int(10), INDEX (messageid), INDEX(personid)) ENGINE=MyISAM");
$dbh->do("CREATE TABLE tf_idf_dictionary (word int(10), messageid int(10), count int(10), INDEX(word), INDEX(messageid), INDEX(count)) ENGINE=MyISAM");
$dbh->do("CREATE TABLE tf_idf_wordmap (word_id int(10), word varchar(255), PRIMARY KEY(word_id), UNIQUE(word)) ENGINE=MyISAM ;");

my @data = `cat $TSV_FILE_NAME`;
splice(@data, 0, 1);

my %user_map = ();
my $next_user_id = 0;


my $file_line = 0;

my %parse_data = ();
my %usercount = ();

print "Reading in tsv\n";
my $messageid = 0;
foreach my $row (@data) {
    # this messageid is an internal counter. Some of these messages won't be saved to the DB so the DB messageid won't
    # necessarily align with these

	
    my @row_fields = split(/\t/, $row);
    if (!(defined($row_fields[$DATETIME_INDEX])) ||
	!(defined($row_fields[$SENDER_INDEX])) ||
	!(defined($row_fields[$TO_LIST_INDEX])) ||
	!(defined($row_fields[$CC_LIST_INDEX])) ||
	!(defined($row_fields[$BCC_LIST_INDEX])) ||
	!(defined($row_fields[$SUBJECT_INDEX])) ||
	!(defined($row_fields[$BODY_INDEX]))) {
	print "skipping malformed line $file_line\n";
	$file_line++;
	next;
    }
    $file_line++;

    my $datetime = $row_fields[$DATETIME_INDEX];
    my $from = $row_fields[$SENDER_INDEX];
    my @to = split(/[,;]/,$row_fields[$TO_LIST_INDEX]);
    my @cc = split(/[,;]/,$row_fields[$CC_LIST_INDEX]);
    my @bcc = split(/[,;]/,$row_fields[$BCC_LIST_INDEX]);
    my $subject = $row_fields[$SUBJECT_INDEX];
    my $body = $row_fields[$BODY_INDEX];
    $body =~ s/\[:newline:\]/\n/g;

    if ($messageid % 1000 == 0) {
	print ".";
    }

    $parse_data{$messageid}{"datetime"} = $datetime;
    $parse_data{$messageid}{"from"} = $from;
    $parse_data{$messageid}{"to"} = \@to;
    $parse_data{$messageid}{"cc"} = \@cc;
    $parse_data{$messageid}{"bcc"} = \@bcc;
    $parse_data{$messageid}{"subject"} = $subject;
    $parse_data{$messageid}{"body"} = $body;

    foreach my $user ((@to, @cc, @bcc)) {
	if (defined $usercount{$user}) {
	    $usercount{$user}++;
	}
	else {
	    $usercount{$user} = 1;
	}
    }
    if (defined $usercount{$from}) {
	$usercount{$from}++;
    }
    else {
      	$usercount{$from} = 1;
    }

    $messageid++;
}
$messageid = undef;

if ($user_limit > 0) {
    print "\nCalculating usercount threshold\n";
    my $usercount_threshold = 1;
    while (scalar keys %usercount > $user_limit) {
	if ($usercount_threshold % 10 == 0) {
	    print ".";
	}

	foreach my $user (keys %usercount) {
	    if ($usercount{$user} < $usercount_threshold) {
		# remove
		delete $usercount{$user};
	    }
	}
	$usercount_threshold++;
    }
    # The actual number of users may be lower than this because later on we filter out emails
    # where the sender has been culled but in doing so some users in the to/cc/bcc fields will
    # also be eliminated
    print "\nUsercount threshold was $usercount_threshold. " . (scalar keys %usercount) . " users remain\n";
}
print "Saving user and message data to database\n";
my $skip_messages = 0;
my $new_messageid = 0;
foreach my $messageid (sort {$a <=> $b} keys %parse_data) {
    #if the sender has been culled, don't save this email
    if (!(defined $usercount{$parse_data{$messageid}{"from"}})) {
	$skip_messages++;
	next;
    }

    $dbh->do("INSERT INTO bodies VALUES($new_messageid, " . $dbh->quote($parse_data{$messageid}{"body"}) . ")");
    $dbh->do("INSERT INTO messages VALUES($new_messageid, " . $dbh->quote($parse_data{$messageid}{"datetime"}) .
	     ", " . getUserIDFromEmail($parse_data{$messageid}{"from"}) . ", " .
	     $dbh->quote($parse_data{$messageid}{"subject"}) . ")");

    insertRecipientFromArray($new_messageid, $parse_data{$messageid}{"to"});
    insertRecipientFromArray($new_messageid, $parse_data{$messageid}{"cc"});
    insertRecipientFromArray($new_messageid, $parse_data{$messageid}{"bcc"});

    $new_messageid++;
}
print "\n";
print "Users: " . (scalar keys %user_map). "\n";
print "Messages: $new_messageid\n";
print "Messages skipped due to lack of sender: $skip_messages\n";
if (defined $dotfidf) {
    my $curdate = `date`;
    print $curdate . "\n";
    setTFIDF();
    $curdate = `date`;
    print $curdate . "\n";
}

sub insertRecipientFromArray($$) {
    (my $new_messageid, my $emailarrayref) = @_;

    my @emailarray = @{$emailarrayref};

    foreach my $email (@emailarray) {
	if (defined $usercount{$email}) {
	    $dbh->do("INSERT INTO recipients VALUES($new_messageid, " . getUserIDFromEmail($email) . ")");
	}
    }
}

sub getUserIDFromEmail($) {
    (my $email) = @_;

    if (defined($user_map{$email})) {
	return $user_map{$email};
    }

    $user_map{$email} = $next_user_id;
    $dbh->do("INSERT INTO people (personid, email) VALUES($next_user_id, " . $dbh->quote($email) . ")");

    $next_user_id++;

    return $user_map{$email};
}


sub setTFIDF() {
    my $sql;
    my $sth;

    print "Processing TFIDF data in memory\n";

    $sql = "SELECT * FROM bodies ORDER BY messageid";
    $sth = $dbh->prepare($sql);
    $sth->execute
	or die "SQL Error: $DBI::errstr\n";
    my %wordcount = ();
    my %emailwords = ();
    my $message_count = scalar @data;

    while (my $row = $sth->fetchrow_hashref) {
	my @body = split(/\s+/,lc($row->{'body'}));

	$stemmer->stem_in_place(\@body);

	my $messageid = $row->{'messageid'};

	if ($messageid % 100 == 0) {
	    print ".";
	}
	if ($messageid % 1000 == 0) {
	    print "$messageid / $message_count\n";
	}
	foreach my $word (@body) {
	    $word =~ s/[\W_]//g;
	    if ($word eq "") {
		next;
	    }
	    if (defined $skip_words{$word}) {
		next;
	    }
	    if (length($word) > 255) {
		next;
	    }

	    #if (defined $wordcount{$word}) {
#		$wordcount{$word}++;#
	    #}
	    #else {
#		$wordcount{$word} = 1;
#	    }

	    if (defined $emailwords{$messageid}{$word}) {
		$emailwords{$messageid}{$word}++;
	    }
	    else {
		$emailwords{$messageid}{$word} = 1;
	    }
	}
    }

    foreach my $messageid (keys %emailwords) {
	foreach my $word (keys %{$emailwords{$messageid}}) {
	    if (defined $wordcount{$word}) {
		$wordcount{$word} += $emailwords{$messageid}{$word};
	    }
	    else {
		$wordcount{$word} = $emailwords{$messageid}{$word};
	    }
	}
    }

    if ($tfidf_wordlimit > 0) {
	print "\nCalculating wordcount threshold\n";
	my $wordcount_threshold = 1;
	while (scalar keys %wordcount > $tfidf_wordlimit) {
	    if ($wordcount_threshold % 10 == 0) {
		print ".";
	    }

	    foreach my $word (keys %wordcount) {
		if ($wordcount{$word} < $wordcount_threshold) {
		    # remove
		    delete $wordcount{$word};
		}
	    }
	    $wordcount_threshold++;
	}
	print "\nWordcount threshold was $wordcount_threshold. " . (scalar keys %wordcount) . " words remain\n";
    }

    print "Writing TFIDF data to database\n";

    # (sort as int not string)
    foreach my $messageid (sort {$a <=> $b} keys %emailwords) {
	if ($messageid % 100 == 0) {
	    print ".";
	}
	if ($messageid % 1000 == 0) {
	    print "$messageid / $message_count\n";
	}
	foreach my $word (keys %{$emailwords{$messageid}}) {
	    # if this is not defined, then we removed it earlier, indicating that we don't want this in the database
	    if (!(defined $wordcount{$word})) {
		next;
	    }

	    my $word_id = getWordID($word);
	    if ($word_id == -1) {
		$dbh->do("INSERT INTO tf_idf_wordmap VALUES($wordIndex, \"$word\")");
		$wordIndex++;
		$word_id = getWordID($word);
	    }
	    $dbh->do("INSERT INTO tf_idf_dictionary VALUES($word_id, $messageid, $emailwords{$messageid}{$word})");
	}
    }
    print "Done writing TFIDF data to database\n";

    $sth->finish();
}

sub getWordID($) {
    (my $word) = @_;

    if (defined $word_id_cache{$word}) {
	return $word_id_cache{$word};
    }

    my $sql = "SELECT word_id FROM tf_idf_wordmap WHERE word=\"$word\"";
    my $sth = $dbh->prepare($sql);
    my $count = $sth->execute
	or die "SQL Error: $DBI::errstr\n";
    if ($count == 0) {
	return -1; # no legitimate words have a dash as they're removed before we insert into the db
    }
    my $row = $sth->fetchrow_hashref;
    my $ret = $row->{'word_id'};
    $sth->finish();

    $word_id_cache{$word} = $ret;

    return $ret;
}

