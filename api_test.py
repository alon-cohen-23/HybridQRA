#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:25:11 2025

@author: aloncohen
"""

import requests


url2 = "http://127.0.0.1:5001/get_collections_names"
response = requests.get(url2)
if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.status_code, response.json())
