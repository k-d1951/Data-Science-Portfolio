# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 02:08:14 2018

"""

import requests

def status_url(code):
    return "https://soa.smext.faa.gov/asws/api/airport/status/" + code

def get_status(code):
    r = requests.get(status_url(code))

    if r.status_code != 200:
            raise Exception ("Error fetching status for airport " + code +    ": <Status: " + str(r.status_code) + ">")
    return r.json()
