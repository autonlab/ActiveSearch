#!/usr/bin/perl -w
use DBI;
use strict;
use Getopt::Long;

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
}


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
		"COL_BODY=i" => \$BODY_INDEX)) {
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

my $messageid = 0;
my $file_line = 0;
foreach my $row (@data) {
	
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

    if ($messageid % 100 == 0) {
	print ".";
    }
    if ($messageid % 1000 == 0) {
	print "$messageid / " . (scalar @data) . "\n";
    }

    $dbh->do("INSERT INTO bodies VALUES($messageid, " . $dbh->quote($body) . ")");
    $dbh->do("INSERT INTO messages VALUES($messageid, " . $dbh->quote($datetime) . ", " . getUserIDFromEmail($from) . ", " . $dbh->quote($subject) . ")");

    insertRecipientFromArray($messageid, \@to);
    insertRecipientFromArray($messageid, \@cc);
    insertRecipientFromArray($messageid, \@bcc);

    $messageid++;
}
print "\n";
print "Users: " . (scalar keys %user_map). "\n";
print "Messages: $messageid\n";
print "Generating TFIDF data\n";
setTFIDF();
#removeTFIDFThreshold(26);

sub insertRecipientFromArray($$) {
    (my $messageid, my $emailarrayref) = @_;

    my @emailarray = @{$emailarrayref};

    foreach my $email (@emailarray) {
	$dbh->do("INSERT INTO recipients VALUES($messageid, " . getUserIDFromEmail($email) . ")");
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

    $sql = "SELECT * FROM bodies ORDER BY messageid";
    $sth = $dbh->prepare($sql);
    $sth->execute
	or die "SQL Error: $DBI::errstr\n";
    while (my $row = $sth->fetchrow_hashref) {
	my %words = ();
	my @body = split(/\s/,lc($row->{'body'}));
	my $messageid = $row->{'messageid'};

	if ($messageid % 100 == 0) {
	    print ".";
	}
	if ($messageid % 1000 == 0) {
	    print "$messageid / " . (scalar @data) . "\n";
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
	    if (defined $words{$word}) {
		$words{$word}++;
	    }
	    else {
		$words{$word} = 1;
	    }
	}

	foreach my $key (keys %words) {
	    my $word_id = getWordID($key);
	    if ($word_id == -1) {
		$dbh->do("INSERT INTO tf_idf_wordmap VALUES($wordIndex, \"$key\")");
		$wordIndex++;
		$word_id = getWordID($key);
	    }
	    $dbh->do("INSERT INTO tf_idf_dictionary VALUES($word_id, $messageid, $words{$key})");
	}
    }
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

sub removeTFIDFThreshold($) {
    (my $count_threshold) = @_;

    # total appearances, not just number of emails appearing in
    print "\n\nDeleting words from tf_idf where the word appears fewer than $count_threshold times\n";

    my $sql = "SELECT word, SUM(count) as cnt FROM tf_idf_dictionary GROUP BY word";
    my $sth = $dbh->prepare($sql);
    my $rows = $sth->execute
	or die "SQL Error: $DBI::errstr\n";
    while (my $row = $sth->fetchrow_hashref) {
	my $word = $row->{'word'};
	if ($row->{'cnt'} < $count_threshold) {
	    $dbh->do("DELETE FROM tf_idf_dictionary WHERE word = $word");
	    $dbh->do("DELETE FROM tf_idf_wordmap WHERE word_id = $word");
	}
	if ($word % 1000 == 0) {
	    print ".";
	}
	if ($word % 10000 == 0) {
	    print "$word / " . $rows . "\n";
	}
    }
    $sth->finish();

    #now compact the word_id so there are no gaps
    print  "compacting wordID numbers so there are no gaps\n";

    my $nextID = 0;
    $sql = "SELECT word_id FROM tf_idf_wordmap ORDER BY word_id";
    $sth = $dbh->prepare($sql);
    $sth->execute
	or die "SQL Error: $DBI::errstr\n";
    while (my $row = $sth->fetchrow_hashref) {
	my $word = $row->{'word_id'};
	# this ID is appropriate where it is
	if ($word == $nextID) {
	    $nextID++;
	}
	# this ID needs to be remapped to nextID
	else {
	    $dbh->do("UPDATE tf_idf_wordmap SET word_id=$nextID WHERE word_id=$word");
	    $dbh->do("UPDATE tf_idf_dictionary SET word=$nextID WHERE word=$word");
	    $nextID++;
	}
	if ($nextID % 100 == 0) {
	    print ".";
	}
    }
    $sth->finish();
}
