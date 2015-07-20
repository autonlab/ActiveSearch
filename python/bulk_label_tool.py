#!/usr/bin/python2.7

import httplib2
import re
import string
import sys

if (len(sys.argv) != 3):
    print "Usage: bulk_label_tool.py <keyword> <label pct as decimal>"
    sys.exit(1)

keyword = sys.argv[1]
label_pct = float(sys.argv[2])

h=httplib2.Http(".cache")
(resp_headers, content) = h.request("http://localhost:5000/getMessagesByKeyword/" + keyword, "GET", headers={'cache-control':'no-cache'})

message_arr = re.split("\n", content)
arg_list = ""
first = 1
count = len(message_arr)
count = count * label_pct

line_count = 0
for line in message_arr:

    line_arr = re.split(" : ", line)
    if (len(line_arr) != 3):
        continue

    msg_id = line_arr[0]

    if (first == 0):
        arg_list += ","
    first = 0

    arg_list += msg_id + ",1"

    (resp_headers, content) = h.request("http://localhost:5000/setLabelBulk/"+msg_id+",1", "GET")
    line_count += 1
    if (line_count > count):
        break


print "set ",line_count, " of ", len(message_arr) , "\n"
