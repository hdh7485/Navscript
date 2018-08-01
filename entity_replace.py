
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

    print('time0 = {}'.format(time1 - time0))
