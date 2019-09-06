# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 02:12:27 2018

"""

import airportstatus, sys, json

if len(sys.argv) < 2:
    print ("Invalid usage")
    exit(1)

print (json.dumps(airportstatus.get_status(sys.argv[1]), indent=2))
