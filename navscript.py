# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import argparse
import sys
import pickle
import time
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

Data_path='dataset/test.txt'

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

if len(sys.argv) > 1:
    lines = sys.argv[1]
else:
    lines =  "What's the weather forecast for this afternoon?"


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)

def load_data(file_name):
    data =  [line.rstrip() for line in list(open(file_name, "r").readlines())]
    x_text = []
    y_script = []
    y_category = []

    for i, sentence in enumerate(data):
        split_result = sentence.split("||")
        x_text.append(split_result[1])
        y_script.append(split_result[3])
        y_category.append(split_result[2])

    return [x_text, y_script, y_category]

one =[
    "What's the WASHINGTON for this Time?",
    "What's the SCHOOL for this Time?",
    "What's the MEETING for this Time?",
    "What's the MONALISA for this Time?",
    "What's the NOTEBOOK for this Time?",
    "What's the SOMETHING for this Time?",

    "Navigate to WASHINGTON.",
    "Navigate to SOMETHING.",
    "Navigate to SCHOOL.",
    "Navigate to MONALISA.",
    "Navigate to DAVID.",

    "What's my WASHINGTON to destination?",
    "What's my SOMETHING to destination?",
    "What's my SCHOOL to destination?",

    "Show me alternative WASHINGTON.",
    "Show me alternative SOMETHING.",
    "Show me alternative SCHOOL.",
    "Show me alternative MONALISA.",

    "Reroute using WASHINGTON.",
    "Reroute using SOMETHING.",
    "Reroute using SCHOOL.",
    "Reroute using MONALISA.",

    "Drive to WASHINGTON.",
    "Drive to SOMETHING.",
    "Drive to SCHOOL.",
    "Drive to MONALISA.",

    "What's my WASHINGTON?",
    "What's my SOMETHING?",
    "What's my SCHOOL?",
    "What's my MONALISA?",

    "Can I make Time's 10am WASHINGTON without recharging?",
    "Can I make Time's 10am SOMETHING without recharging?",
    "Can I make Time's 10am SCHOOL without recharging?",
    "Can I make Time's 10am MEETING without recharging?",

    "Will it rain Time in WASHINGTON?",
    "Will it rain Time in SOMETHING?",
    "Will it rain Time in SCHOOL?",
    "Will it rain Time in MEETING?",

    "How long can I go?",
    "How far I can go?",
    "How much longer can I go?",

    "What's traffic like on the WASHINGTON?",
    "What's traffic like on the SOMETHING?",
    "What's traffic like on the SCHOOL?",
    "What's traffic like on the MONALISA?"
]
one_script = [
    "[SEARCH FROM:WASHINGTON  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:SCHOOL  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:MEETING  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:MONALISA  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:NOTEBOOK  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:SOMETHING  WHERE:HERE WHEN:Time]",

    "[ROUTE TO:[SEARCH KEYWORD:WASHINGTON]]",
    "[ROUTE TO:[SEARCH KEYWORD:SOMETHING]]",
    "[ROUTE TO:[SEARCH KEYWORD:SCHOOL]]",
    "[ROUTE TO:[SEARCH KEYWORD:MONALISA]]",
    "[ROUTE TO:[SEARCH KEYWORD:DAVID]]",

    "[ROUTE INFO:WASHINGTON]",
    "[ROUTE INFO:SOMETHING]",
    "[ROUTE INFO:SCHOOL]",

    "[ROUTE WASHINGTON]",
    "[ROUTE SOMETHING]",
    "[ROUTE SCHOOL]",
    "[ROUTE MONALISA]",

    "[ROUTE ALTROUTE USE:[SEARCH LINKS:WASHINGTON]]",
    "[ROUTE ALTROUTE USE:[SEARCH LINKS:SOMETHING]]",
    "[ROUTE ALTROUTE USE:[SEARCH LINKS:SCHOOL]]",
    "[ROUTE ALTROUTE USE:[SEARCH LINKS:MONALISA]]",

    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:WASHINGTON]]]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:SOMETHING]]]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:SCHOOL]]]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:MONALISA]]",

    "[MODE WASHINGTON]",
    "[MODE SOMETHING]",
    "[MODE SCHOOL]",
    "[MODE MONALISA]",

    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:WASHINGTON FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:SOMETHING FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:SCHOOL FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:MEETING FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",

    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:WASHINGTON] WHEN:Time]",
    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:SOMETHING] WHEN:Time]",
    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:SCHOOL] WHEN:Time]",
    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:MEETING] WHEN:Time]",

    "[MODE DRIVERANGE]",
    "[MODE DRIVERANGE]",
    "[MODE DRIVERANGE]",

    "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",


]
one_class_id=[
    0,0,0,0,0,0,
    8,8,8,8,8,
    15,15,15,
    16,16,16,16,
    9,9,9,9,
    10,10,10,10,
    18,18,18,18,
    11,11,11,11,
    14,14,14,14,
    18,18,18,
    12,12,12,12
]
two = [
    "What's the WASHINGTON like on my WASHINGTON?",
    "What's the WASHINGTON like on my SOMETHING?",
    "What's the SOMETHING like on my WASHINGTON?",
    "What's the SOMETHING like on my SOMETHING?",

    "Can you find me a WASHINGTON with WASHINGTON nearby?",
    "Can you find me a WASHINGTON with SOMETHING nearby?",
    "Can you find me a WASHINGTON with SCHOOL nearby?",
    "Can you find me a SOMETHING with WASHINGTON nearby?",
    "Can you find me a SOMETHING with SOMETHING nearby?",
    "Can you find me a SOMETHING with SCHOOL nearby?",
    "Can you find me a SCHOOL with WASHINGTON nearby?",
    "Can you find me a SCHOOL with SOMETHING nearby?",
    "Can you find me a SCHOOL with SCHOOL nearby?",

    "Find a WASHINGTON along WASHINGTON.",
    "Find a WASHINGTON along SOMETHING.",
    "Find a WASHINGTON along SCHOOL.",
    "Find a SOMETHING along WASHINGTON.",
    "Find a SOMETHING along SOMETHING.",
    "Find a SOMETHING along SCHOOL.",
    "Find a SCHOOL along WASHINGTON.",
    "Find a SCHOOL along SOMETHING.",
    "Find a SCHOOL along SCHOOL.",

    "Find the cheapest indoor WASHINGTON within 500 meters of my WASHINGTON.",
    "Find the cheapest indoor WASHINGTON within 500 meters of my SOMETHING.",
    "Find the cheapest indoor WASHINGTON within 500 meters of my SCHOOL.",
    "Find the cheapest indoor SOMETHING within 500 meters of my WASHINGTON.",
    "Find the cheapest indoor SOMETHING within 500 meters of my SOMETHING.",
    "Find the cheapest indoor SOMETHING within 500 meters of my SCHOOL.",
    "Find the cheapest indoor SCHOOL within 500 meters of my WASHINGTON.",
    "Find the cheapest indoor SCHOOL within 500 meters of my SOMETHING.",
    "Find the cheapest indoor SCHOOL within 500 meters of my SCHOOL.",

    # "What's WASHINGTON like on the WASHINGTON?",
    # "What's WASHINGTON like on the SOMETHING?",
    # "What's WASHINGTON like on the SCHOOL?",
    # "What's WASHINGTON like on the MONALISA?",
    #
    # "What's SOMETHING like on the WASHINGTON?",
    # "What's SOMETHING like on the SOMETHING?",
    # "What's SOMETHING like on the SCHOOL?",
    # "What's SOMETHING like on the MONALISA?",
    #
    # "What's SCHOOL like on the WASHINGTON?",
    # "What's SCHOOL like on the SOMETHING?",
    # "What's SCHOOL like on the SCHOOL?",
    # "What's SCHOOL like on the MONALISA?",
    #
    # "What's MONALISA like on the WASHINGTON?",
    # "What's MONALISA like on the SOMETHING?",
    # "What's MONALISA like on the SCHOOL?",
    # "What's MONALISA like on the MONALISA?",

    "Are there any WASHINGTON on my WASHINGTON?",
    "Are there any WASHINGTON on my SOMETHING?",
    "Are there any WASHINGTON on my SCHOOL?",
    "Are there any WASHINGTON on my MONALISA?",
    "Are there any WASHINGTON on my NOTEBOOK?",

    "Are there any SOMETHING on my WASHINGTON?",
    "Are there any SOMETHING on my SOMETHING?",
    "Are there any SOMETHING on my SCHOOL?",
    "Are there any SOMETHING on my MONALISA?",
    "Are there any SOMETHING on my NOTEBOOK?",

    "Are there any SCHOOL on my WASHINGTON?",
    "Are there any SCHOOL on my SOMETHING?",
    "Are there any SCHOOL on my SCHOOL?",
    "Are there any SCHOOL on my MONALISA?",
    "Are there any SCHOOL on my NOTEBOOK?",

    "Are there any MONALISA on my WASHINGTON?",
    "Are there any MONALISA on my SOMETHING?",
    "Are there any MONALISA on my SCHOOL?",
    "Are there any MONALISA on my MONALISA?",
    "Are there any MONALISA on my NOTEBOOK?",

    "Are there any NOTEBOOK on my WASHINGTON?",
    "Are there any NOTEBOOK on my SOMETHING?",
    "Are there any NOTEBOOK on my SCHOOL?",
    "Are there any NOTEBOOK on my MONALISA?",
    "Are there any NOTEBOOK on my NOTEBOOK?"

]
two_script = [
    "[SEARCH FROM:WASHINGTON  WHERE:WASHINGTON]",
    "[SEARCH FROM:WASHINGTON  WHERE:SOMETHING]",
    "[SEARCH FROM:SOMETHING WHERE:WASHINGTON]",
    "[SEARCH FROM:SOMETHING WHERE:SOMETHING]",

    "[SEARCH FROM:WASHINGTON WHERE:NEARBY WITH:WASHINGTON]",
    "[SEARCH FROM:WASHINGTON WHERE:NEARBY WITH:SOMETHING]",
    "[SEARCH FROM:WASHINGTON WHERE:NEARBY WITH:RESTROOM]",
    "[SEARCH FROM:SOMETHING WHERE:NEARBY WITH:WASHINGTON]",
    "[SEARCH FROM:SOMETHING WHERE:NEARBY WITH:SOMETHING]",
    "[SEARCH FROM:SOMETHING WHERE:NEARBY WITH:SCHOOL]",
    "[SEARCH FROM:SCHOOL WHERE:NEARBY WITH:WASHINGTON]",
    "[SEARCH FROM:SCHOOL WHERE:NEARBY WITH:SOMETHING]",
    "[SEARCH FROM:SCHOOL WHERE:NEARBY WITH:SCHOOL]",

    "[SEARCH ONE FROM:WASHINGTON WHERE:WASHINGTON]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SOMETHING]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SCHOOL]",
    "[SEARCH ONE FROM:SOMETHING WHERE:WASHINGTON]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SOMETHING]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SCHOOL]",
    "[SEARCH ONE FROM:SCHOOL WHERE:WASHINGTON]",
    "[SEARCH ONE FROM:SCHOOLWHERE:SOMETHING]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SCHOOL]",

    "[SEARCH ONE FROM:WASHINGTON WHERE:WASHINGTON RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SOMETHING RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SCHOOL RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SOMETHING WHERE:WASHINGTON RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SOMETHING RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SCHOOL RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SCHOOL WHERE:WASHINGTON RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SOMETHING RANGE:500M WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SCHOOL RANGE:500M WITH:[SORT PRICE ASC]]",

    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    #
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    #
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    #
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",

    "[MODE WASHINGTON WHERE:WASHINGTON WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON WHERE:SOMETHING WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON WHERE:SCHOOL WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON WHERE:MONALISA WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON WHERE:NOTEBOOK WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE SOMETHING WHERE:WASHINGTON WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING WHERE:SOMETHING WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING WHERE:SCHOOL WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING WHERE:MONALISA WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING WHERE:NOTEBOOK WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE SCHOOL WHERE:WASHINGTON WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL WHERE:SOMETHING WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL WHERE:SCHOOL WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL WHERE:MONALISA WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL WHERE:NOTEBOOK WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE MONALISA WHERE:WASHINGTON WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA WHERE:SOMETHING WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA WHERE:SCHOOL WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA WHERE:MONALISA WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA WHERE:NOTEBOOK WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE NOTEBOOK WHERE:WASHINGTON WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK WHERE:SOMETHING WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK WHERE:SCHOOL WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK WHERE:MONALISA WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK WHERE:NOTEBOOK WITH:[VOICERESPONSE TEMPLATE:””*]]"
]
two_class_id=[
    1, 1, 1, 1,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13

]
three = [
    "Show me a WASHINGTON on WASHINGTON and WASHINGTON.",
    "Show me a WASHINGTON on WASHINGTON and SOMETHING.",
    "Show me a WASHINGTON on WASHINGTON and SCHOOL.",
    "Show me a WASHINGTON on SOMETHING and WASHINGTON.",
    "Show me a WASHINGTON on SOMETHING and SCHOOL.",
    "Show me a WASHINGTON on SOMETHING and SOMETHING.",
    "Show me a WASHINGTON on SCHOOL and WASHINGTON.",
    "Show me a WASHINGTON on SCHOOL and SOMETHING.",
    "Show me a WASHINGTON on SCHOOL and SCHOOL.",

    "Show me a SOMETHING on WASHINGTON and WASHINGTON.",
    "Show me a SOMETHING on WASHINGTON and SOMETHING.",
    "Show me a SOMETHING on WASHINGTON and SCHOOL.",
    "Show me a SOMETHING on SOMETHING and WASHINGTON.",
    "Show me a SOMETHING on SOMETHING and SCHOOL.",
    "Show me a SOMETHING on SOMETHING and SOMETHING.",
    "Show me a SOMETHING on SCHOOL and WASHINGTON.",
    "Show me a SOMETHING on SCHOOL and SOMETHING.",
    "Show me a SOMETHING on SCHOOL and SCHOOL.",

    "Show me a SCHOOL on WASHINGTON and WASHINGTON.",
    "Show me a SCHOOL on WASHINGTON and SOMETHING.",
    "Show me a SCHOOL on WASHINGTON and SCHOOL.",
    "Show me a SCHOOL on SOMETHING and WASHINGTON.",
    "Show me a SCHOOL on SOMETHING and SCHOOL.",
    "Show me a SCHOOL on SOMETHING and SOMETHING.",
    "Show me a SCHOOL on SCHOOL and WASHINGTON.",
    "Show me a SCHOOL on SCHOOL and SOMETHING.",
    "Show me a SCHOOL on SCHOOL and SCHOOL.",

    "Okay, can you find me a WASHINGTON on my WASHINGTON that has a WASHINGTON?",
    "Okay, can you find me a WASHINGTON on my WASHINGTON that has a SOMETHING?",
    "Okay, can you find me a WASHINGTON on my WASHINGTON that has a SCHOOL?",
    "Okay, can you find me a WASHINGTON on my SOMETHING that has a WASHINGTON?",
    "Okay, can you find me a WASHINGTON on my SOMETHING that has a SOMETHING?",
    "Okay, can you find me a WASHINGTON on my SOMETHING that has a SCHOOL?",
    "Okay, can you find me a WASHINGTON on my SCHOOL that has a WASHINGTON?",
    "Okay, can you find me a WASHINGTON on my SCHOOL that has a SOMETHING?",
    "Okay, can you find me a WASHINGTON on my SCHOOL that has a SCHOOL?",

    "Okay, can you find me a SOMETHING on my WASHINGTON that has a WASHINGTON?",
    "Okay, can you find me a SOMETHING on my WASHINGTON that has a SOMETHING?",
    "Okay, can you find me a SOMETHING on my WASHINGTON that has a SCHOOL?",
    "Okay, can you find me a SOMETHING on my SOMETHING that has a WASHINGTON?",
    "Okay, can you find me a SOMETHING on my SOMETHING that has a SOMETHING?",
    "Okay, can you find me a SOMETHING on my SOMETHING that has a SCHOOL?",
    "Okay, can you find me a SOMETHING on my SCHOOL that has a WASHINGTON?",
    "Okay, can you find me a SOMETHING on my SCHOOL that has a SOMETHING?",
    "Okay, can you find me a SOMETHING on my SCHOOL that has a SCHOOL?",

    "Okay, can you find me a SCHOOL on my WASHINGTON that has a WASHINGTON?",
    "Okay, can you find me a SCHOOL on my WASHINGTON that has a SOMETHING?",
    "Okay, can you find me a SCHOOL on my WASHINGTON that has a SCHOOL?",
    "Okay, can you find me a SCHOOL on my SOMETHING that has a WASHINGTON?",
    "Okay, can you find me a SCHOOL on my SOMETHING that has a SOMETHING?",
    "Okay, can you find me a SCHOOL on my SOMETHING that has a SCHOOL?",
    "Okay, can you find me a SCHOOL on my SCHOOL that has a WASHINGTON?",
    "Okay, can you find me a SCHOOL on my SCHOOL that has a SOMETHING?",
    "Okay, can you find me a SCHOOL on my SCHOOL that has a SCHOOL?",

    "Find WASHINGTON near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find WASHINGTON near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find WASHINGTON near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find WASHINGTON near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find WASHINGTON near destination that accepts SOMETHING and has a SOMETHING.",
    "Find WASHINGTON near destination that accepts SOMETHING and has a SCHOOL.",
    "Find WASHINGTON near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find WASHINGTON near destination that accepts SCHOOL and has a SOMETHING.",
    "Find WASHINGTON near destination that accepts SCHOOL and has a SCHOOL.",

    "Find SOMETHING near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find SOMETHING near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find SOMETHING near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find SOMETHING near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find SOMETHING near destination that accepts SOMETHING and has a SOMETHING.",
    "Find SOMETHING near destination that accepts SOMETHING and has a SCHOOL.",
    "Find SOMETHING near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find SOMETHING near destination that accepts SCHOOL and has a SOMETHING.",
    "Find SOMETHING near destination that accepts SCHOOL and has a SCHOOL.",

    "Find SCHOOL near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find SCHOOL near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find SCHOOL near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find SCHOOL near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find SCHOOL near destination that accepts SOMETHING and has a SOMETHING.",
    "Find SCHOOL near destination that accepts SOMETHING and has a SCHOOL.",
    "Find SCHOOL near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find SCHOOL near destination that accepts SCHOOL and has a SOMETHING.",
    "Find SCHOOL near destination that accepts SCHOOL and has a SCHOOL.",

    "Find MONALISA near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find MONALISA near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find MONALISA near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find MONALISA near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find MONALISA near destination that accepts SOMETHING and has a SOMETHING.",
    "Find MONALISA near destination that accepts SOMETHING and has a SCHOOL.",
    "Find MONALISA near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find MONALISA near destination that accepts SCHOOL and has a SOMETHING.",
    "Find MONALISA near destination that accepts SCHOOL and has a SCHOOL.",

    "Find NOTEBOOK near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find NOTEBOOK near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find NOTEBOOK near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find NOTEBOOK near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find NOTEBOOK near destination that accepts SOMETHING and has a SOMETHING.",
    "Find NOTEBOOK near destination that accepts SOMETHING and has a SCHOOL.",
    "Find NOTEBOOK near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find NOTEBOOK near destination that accepts SCHOOL and has a SOMETHING.",
    "Find NOTEBOOK near destination that accepts SCHOOL and has a SCHOOL."
]
three_script = [
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and WASHINGTON]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and SOMETHING]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and SCHOOL]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and WASHINGTON]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and SCHOOL]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and SOMETHING]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and WASHINGTON]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and SOMETHING]]",
    "[SEARCH FROM:WASHINGTON  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and SCHOOL]]",

    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and WASHINGTON]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and SOMETHING]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and SCHOOL]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and WASHINGTON]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and SCHOOL]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and SOMETHING]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and WASHINGTON]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and SOMETHING]]",
    "[SEARCH FROM:SOMETHING  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and SCHOOL]]",

    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and WASHINGTON]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and SOMETHING]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON and SCHOOL]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and WASHINGTON]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and SCHOOL]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:SOMETHING and SOMETHING]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and WASHINGTON]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and SOMETHING]]",
    "[SEARCH FROM:SCHOOL  WHERE:[SEARCH GEOCODE WHERE:SCHOOL and SCHOOL]]",

    "[SEARCH ONE FROM:WASHINGTON WHERE:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:WASHINGTON WHERE:SCHOOL WITH:SCHOOL]",

    "[SEARCH ONE FROM:SOMETHING WHERE:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SOMETHING WHERE:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:SOMETHING WHERE:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:SOMETHING WHERE:SCHOOL WITH:SCHOOL]",

    "[SEARCH ONE FROM:SCHOOL WHERE:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SCHOOL WHERE:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:SCHOOL WHERE:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:SCHOOL WHERE:SCHOOL WITH:SCHOOL]",

    "[SEARCH ONE FROM:WASHINGTON WITH:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:WASHINGTON WITH:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:WASHINGTON WITH:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:WASHINGTON WITH:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:WASHINGTON WITH:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:WASHINGTON WITH:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:WASHINGTON WITH:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:WASHINGTON WITH:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:WASHINGTON WITH:SCHOOL WITH:SCHOOL]",

    "[SEARCH ONE FROM:SOMETHING WITH:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SOMETHING WITH:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:SOMETHING WITH:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:SOMETHING WITH:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SOMETHING WITH:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:SOMETHING WITH:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:SOMETHING WITH:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SOMETHING WITH:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:SOMETHING WITH:SCHOOL WITH:SCHOOL]",

    "[SEARCH ONE FROM:SCHOOL WITH:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SCHOOL WITH:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:SCHOOL WITH:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:SCHOOL WITH:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SCHOOL WITH:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:SCHOOL WITH:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:SCHOOL WITH:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:SCHOOL WITH:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:SCHOOL WITH:SCHOOL WITH:SCHOOL]",

    "[SEARCH ONE FROM:MONALISA WITH:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:MONALISA WITH:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:MONALISA WITH:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:MONALISA WITH:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:MONALISA WITH:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:MONALISA WITH:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:MONALISA WITH:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:MONALISA WITH:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:MONALISA WITH:SCHOOL WITH:SCHOOL]",

    "[SEARCH ONE FROM:NOTEBOOK WITH:WASHINGTON WITH:WASHINGTON]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:WASHINGTON WITH:SOMETHING]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:WASHINGTON WITH:SCHOOL]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:SOMETHING WITH:WASHINGTON]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:SOMETHING WITH:SOMETHING]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:SOMETHING WITH:SCHOOL]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:SCHOOL WITH:WASHINGTON]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:SCHOOL WITH:SOMETHING]",
    "[SEARCH ONE FROM:NOTEBOOK WITH:SCHOOL WITH:SCHOOL]"
]
three_class_id=[
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7

]

message_total = [one,two,three]
message_embeddings_total=[]
embeded_class_id=[one_class_id,two_class_id,three_class_id]

for i in range(len(message_total)):
    print(len(message_total[i]))
    print(len(embeded_class_id[i]))
    assert len(message_total[i]) !=embeded_class_id[i]

script_total = [one_script, two_script, three_script]

google_entity_type ={0:'UNKNOWN', 1:'PERSON', 2:'LOCATION', 3:'ORGANIZATION', 4:'EVENT', 5:'WORK_OF_ART', 6:'CONSUMER_GOOD', 7:'OTHER'}
entity_type ={0:'X', 1:'DAVID', 2:'WASHINGTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
i_entity_type ={'X':0, 'DAVID':1, 'WASHINGTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}
Times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
line_time=''

Correct_sentence=[]
Correct_script=[]
Correct_classID=[]
Correct_answer=[]
Correct_ID=[]
Correct_navscript=[]
Wrong_sentence=[]
Wrong_script=[]
Wrong_classID=[]
Wrong_answer=[]
Wrong_ID=[]
Wrong_navscript=[]

module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
embed = hub.Module(module_url)
messages = tf.placeholder(dtype=tf.string, shape=[None])
embedding = embed(messages)

session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
t_sentences, t_scripts, class_ID = load_data(Data_path)
print("start embed")
# Import the Universal Sentence Encoder's TF Hub module
print("end")
for test_enum, Input_data in enumerate(zip(t_sentences,t_scripts,class_ID)):
    time0 = time.time()
    lines=Input_data[0]
    for i in range(len(Times)):
        if Times[i] in lines:
            line_time = Times[i]
            lines = lines.replace(Times[i], 'Time')
    light_module = True
    client = language.LanguageServiceClient()
    document = types.Document(
        content=lines,
        language='en',
        type=enums.Document.Type.PLAIN_TEXT)
    # document = types.Document(content=lines,language='en',type=enums.Document.Type.PLAIN_TEXT)
    entities = client.analyze_entities(document).entities

    result = Input_data[0]

    for entity in entities:
        result = result.replace(entity.name, entity_type[entity.type])




    print("\n\ninput : {}".format(Input_data))
    print("Replace nouns: {}\n\n".format(result))

    time1 = time.time()
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
    # print("start embed")
    # # Import the Universal Sentence Encoder's TF Hub module
    # embed = hub.Module(module_url)
    #
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)


    if not os.path.exists("./profile2.bin"):
        print("There is no profile2.bin. Making profile2.bin")
        for message in message_total:
            message_embeddings = session.run(embed(message))
            message_embeddings_total.append(message_embeddings)
        with open('profile2.bin', 'wb') as f:
            pickle.dump(message_embeddings_total, f)
        print("Finish make profile!")
    else:
        print("profile2.bin exists")
        with open('./profile2.bin', 'rb') as f:
            message_embeddings_total = pickle.load(f)

    print("\n\ncompare with : "+ str(len(message_total[0]))+", "+str(len(message_total[1]))+", "+str(len(message_total[2])))
    print("compare with : "+ str(len(message_embeddings_total[0]))+", "+str(len(message_embeddings_total[1]))+", "+str(len(message_embeddings_total[2])))


    time2 = time.time()
    test_message_embeddings = session.run(embedding, feed_dict={messages: [result]})
    time3 = time.time()

    minimum = 100
    minimum_index = 0
    entity_num = len(entities)
    if entity_num==0:
        entity_num=1
    # for i, message_embedding in enumerate(message_embeddings):
    #     error = rmse(np.array(message_embedding), np.array(test_message_embeddings))
    #     if minimum > error:
    #       minimum = error
    #       minimum_index = i

    for i, message_embedding in enumerate(message_embeddings_total[entity_num-1]):
        error = rmse(np.array(message_embedding), np.array(test_message_embeddings))
        if minimum > error:
          minimum = error
          minimum_index = i


    print("\n\n\n\n\n")


    print("Minimum RMSE value: {}".format(minimum))
    print("Most similar script: {}".format(script_total[entity_num-1][minimum_index]))
    print("Estimation: {}".format(minimum_index))
    #print("Answer: {}\n".format(test_label))
    result2 = script_total[entity_num-1][minimum_index] #query

    Dict_entitis={}
    K=[] #Keys = Type
    V=[] #Values = Names
    google=[]
    for entity in entities:
        # Dict_entitis[entity.name]=entity_type[entity.type]
        google.append(google_entity_type[entity.type])
        K.append(entity_type[entity.type])
        V.append(entity.name)


    # print("Dict_entitis : ", Dict_entitis)
    print("Entities : ", google)
    print("Keys : ", K)
    print("Values : ", V)
    print("entity_number : ", entity_num)

    # for key, value in Dict_entitis.items():
    #     result2 = result2.replace(value, key)

    number = len(K)
    for i in range(0, number):
        result2 = result2.replace(K[i], V[i],1)

    if line_time is not '':
        result2 = result2.replace('Time', line_time,1)

    print("input: {}".format(Input_data[0]))
    print("Replace nouns: {}".format(result))
    print("Selected Sentence: {}".format(message_total[entity_num-1][minimum_index]))
    print("Query: {}".format(result2))

    if embeded_class_id[entity_num-1][minimum_index] != Input_data[2]:
        Wrong_sentence = Wrong_sentence + [Input_data[0]]
        Wrong_script = Wrong_sentence + [Input_data[1]]
        Wrong_classID=Wrong_classID+[embeded_class_id[entity_num-1][minimum_index]]
        Wrong_answer = Wrong_answer + [Input_data[2]]
        if result2.upper() != Input_data[1].upper():
            Wrong_navscript=Wrong_navscript+["X"]
        else:
            Wrong_navscript = Wrong_navscript + ["O"]
    else:
        Correct_sentence = Correct_sentence + [Input_data[0]]
        Correct_script = Correct_script + [Input_data[1]]
        Correct_classID = Correct_classID + [embeded_class_id[entity_num - 1][minimum_index]]
        Correct_answer = Correct_answer + [Input_data[2]]
        if result2.upper() != Input_data[1].upper():
            Correct_navscript=Correct_navscript+["X"]
        else:
            Correct_navscript = Correct_navscript + ["O"]
    time4 = time.time()

    print('time0 = {}'.format(time1 - time0))
    print('time1 = {}'.format(time2 - time1))
    print('time2 = {}'.format(time3 - time2))
    print('time3 = {}'.format(time4 - time3))
    print('total time = {}'.format(time4 - time0))

w=open("Result.txt",'w')
for enum, X in enumerate(zip(Correct_answer,Correct_classID, Correct_script, Correct_navscript)):
    w.write(str(X[0])+"||"+str(X[1])+"||"+str(X[2])+"||"+"O"+"||"+str(X[3]))
for enum, X in enumerate(zip(Wrong_answer,Wrong_classID, Wrong_script, Wrong_navscript)):
    w.write(str(X[0])+"||"+str(X[1])+"||"+str(X[2])+"||"+"X"+"||"+str(X[3]))
