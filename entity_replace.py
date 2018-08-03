# -*- coding:utf-8 -*-
import time
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

entity_type ={0:'X', 1:'DAVID', 2:'WASHINGTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
i_entity_type ={'X':0, 'DAVID':1, 'WASHINGTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}
distance = ['meters']
line_time=''

def find_and_change_entity(input_sentence):
    entity_counter = [0, 0, 0, 0, 0, 0, 0, 0]
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
        replaced_sentence = input_sentence.replace(entity.name, entity_type[entity.type] + str(entity_counter[entity.type]))
        replace_save_dict.update({entity.name:entity_type[entity.type]+str(entity_counter[entity.type])})
        entity_counter[entity.type] += 1

    print("Input: {}".format(input_sentence))
    print("Find entities: {}".format(entities))
    print("Replace dictionary: {}".format(replace_save_dict))
    print("Replaced sentence: {}\n".format(replaced_sentence))

    return replaced_sentence, replace_save_dict

def replace_to_script(input_script, replace_save_dict):
    for replace_element in replace_save_dict:
        input_script = input_script.replace(replace_save_dict[replace_element], replace_element)
    replaced_script = input_script
    print("Input_script: {}".format(input_script))
    print("Input_save_dict: {}".format(replace_save_dict))
    print("Replaced script: {}".format(replaced_script))
    return replaced_script

def main():
    input_sentence = "Can you find me a gas station with restroom facilities nearby?"
    skeleton_script = "[SEARCH FROM:WASHINGTON0 WHERE:NEARBY WITH:WASHINGTON1]"

    time0 = time.time()
    entity_changed_sentence, replace_saved_dict = find_and_change_entity(input_sentence)
    time1 = time.time()
    result = replace_to_script(skeleton_script, replace_saved_dict)
    time2 = time.time()
    print("run time0: {}".format(time1 - time0))
    print("run time1: {}".format(time2 - time1))
    print("total run time: {}".format(time2 - time0))
    print("result: {}".format(result))

if __name__ == '__main__':
    main()
