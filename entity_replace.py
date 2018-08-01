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
import pprint
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

google_entity_type ={0:'UNKNOWN', 1:'PERSON', 2:'LOCATION', 3:'ORGANIZATION', 4:'EVENT', 5:'WORK_OF_ART', 6:'CONSUMER_GOOD', 7:'OTHER'}
entity_type ={0:'X', 1:'DAVID', 2:'WASHINGTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
i_entity_type ={'X':0, 'DAVID':1, 'WASHINGTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}
distance = ['meters']
line_time=''

def find_and_change_entity(input_sentence):
    #Preprocess: Find time domain words.
    times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    for i in range(len(times)):
        if times[i] in input_sentence:
            line_time = times[i]
            input_sentence = input_sentence.replace(times[i], 'Time')

    client = language.LanguageServiceClient()
    document = types.Document(
        content=input_sentence,
        language='en',
        type=enums.Document.Type.PLAIN_TEXT)

    entities = client.analyze_entities(document).entities

    replace_save_dict = {}
    for entity in entities:
        replaced_sentence = input_sentence.replace(entity.name, entity_type[entity.type])
        replace_save_dict.update({entity.name:entity_type[entity.type]})

    print("Input: {}".format(input_sentence))
    print("Find entities: {}".format(entities))
    pprint.pprint("Replace dictionary: {}".format(replace_save_dict))
    print("Replaced nouns: {}\n".format(replaced_sentence))

    return replaced_sentence, replace_save_dict

def replace_to_script(input_script, replace_save_dict):
    for replace_element in replace_save_dict:
        
    replaced_script = "qwer" 
    print("Input_script: {}".format(input_script))
    pprint.pprint("Input_save_dict: {}".format(replace_save_dict))
    print("Replaced script: {}".format(replaced_script))
    return replaced_script

def main():
    input_sentence = "Can you find me a gas station with restroom facilities nearby?"
    output_script = "[SEARCH FROM:GASSTATION WHERE:NEARBY WITH:RESTROOM]"

    entity_changed_sentence, replace_saved_dict = find_and_change_entity(input_sentence)
    replace_to_script(entity_changed_sentence, replace_saved_dict)

if __name__ == '__main__':
    main()
