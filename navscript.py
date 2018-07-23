import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
import os
import pickle
import time
import data_loader
from pathlib import Path
import sentencepiece as spm

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

start_time = time.time()
result_sheet_file = open('./result_sheet.txt', 'w')
result_sheet_file.write('enum||input||answer||estimate||correct\n')
result_sheet_file.close()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

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

tf.logging.set_verbosity(tf.logging.ERROR)
light_module = False

#light_module = True

if len(sys.argv) > 1:
    lines = [sys.argv[1]]
else:
    loaded_data = data_loader.load_data('./dataset/test.txt')
    lines = loaded_data[0]
    y = loaded_data[1]
    answer_correct = 0 

scripts = ["[SEARCH FROM:SOMETHING1 WHERE:HERE WHEN:TIME1]",
        "[SEARCH FROM:SOMETHING1 WHERE:SOMETHING2]",
        "[SEARCH FROM:SOMETHING1 WHERE:[SEARCH GEOCODE WHERE:PLACE1]]",
        "[SEARCH FROM:SCHOOL WHERE:NEARBY WITH:SOMETHING1]",
        "[SEARCH ONE FROM:PLACE1 WHERE:PLACE2]",
        "[SEARCH ONE FROM:SOMETHING1 WHERE:SOMETHING2 RANGE:500M WITH:[SORT PRICE ASC]]",
        "[SEARCH ONE FROM:PLACE1 WHERE:SOMETHING1 WITH:PLACE2]",
        "[SEARCH ONE FROM:SOMETHING1 WITH:SOMETHING2 WITH:PLACE1]",
        "[ROUTE TO:[SEARCH KEYWORD:PLACE1]]",
        "[ROUTE INFO:SOMETHING1]",
        "[ROUTE ALTROUTE]",
        "[ROUTE ALTROUTE USE:[SEARCH LINKS:SOMETHING1]]",

        "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:PLACE1]]]",
        "[MODE SOMETHING1]",
        "[MODE SOMETHING1 TO:[SEARCH KEYWORD:MEETTING FROM:SCHEDULE WHEN:TIME1] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
        "[MODE SOMETHING1 [SEARCH FROM:TRAFFIC WHERE:[SEARCH KEYWORD:PLACE1]] WITH:[VOICERESPONSE TEMPLATE:""*]",
        "[MODE SOMETHING1 WHERE:SOMETHING2 WITH:[VOICERESPONSE TEMPLATE:""*]]",
        "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:PLACE1] WHEN:TIME1]"
        ]

messages2 = [line.rstrip('\n') for line in open('profile_messages.txt')]
#print(messages2)

if light_module:
    #with tf.Session() as sess:
    #    spm_path = sess.run(embed(signature="spm_path"))
    spm_path = module_url + '/assets/universal_encoder_8k_spm.model'
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    print("SentencePiece model loaded at {}.".format(spm_path))

    input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
    encodings = embed(
            inputs=dict(
                values=input_placeholder.values,
                indices=input_placeholder.indices,
                dense_shape=input_placeholder.dense_shape))

if light_module:
    if not os.path.exists("./profile_lite.bin"):
        print("There is no profile_lite.bin. Making profile_lite.bin")
        values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages2)
        # Reduce logging output.

        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(
                encodings,
                feed_dict={input_placeholder.values: values,
                    input_placeholder.indices: indices,
                    input_placeholder.dense_shape: dense_shape})
        with open('profile_lite.bin', 'wb') as f:
            pickle.dump(message_embeddings, f)
        print("Finish make profile!")
    else:
        print("profile_lite.bin exists")
        with open('./profile_lite.bin', 'rb') as f:
            message_embeddings = pickle.load(f)
else:
    if not os.path.exists("./profile.bin"):
        print("There is no profile.bin. Making profile.bin")
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(messages2))
        with open('profile.bin', 'wb') as f:
            pickle.dump(message_embeddings, f)
        print("Finish make profile!")
    else:
        print("profile.bin exists")
        with open('./profile.bin', 'rb') as f:
            message_embeddings = pickle.load(f)
#make_bin_time = time.time()

if light_module:
    #sentence-encoder-light/2
    #module_url =  "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
    module_url = "./modules/539544f0a997d91c327c23285ea00c37588d92cc"
else:
    #sentence-encoder/2
    #module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
    module_url = "./modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"
    #sentence-encoder/1
    #module_url = "/Users/hdh7485/navscript/modules/c6f5954ffa065cdb2f2e604e740e8838bf21a2d3"

# Import the Universal Sentence Encoder's TF Hub module

embed = hub.Module(module_url)
messages = tf.placeholder(dtype=tf.string, shape=[None])
embedding = embed(messages)

session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])

for test_enum, x_text in enumerate(lines):
    print('Input: {}'.format(x_text ))

    unknow_pool = ['SOMETHING1', 'SOMETHING2', 'SOMETHING3', 'SOMETHING4', 'SOMETHING5']
    person_pool = ['DAVID', 'PACTRICK', 'BOB', 'HARRY', 'ERIC']
    location_pool = ['PLACE1', 'PLACE2', 'PLACE3', 'PLACE4', 'PLACE5']
    #location_pool = ['WASHINGTON', 'SEOUL', 'MADRID', 'LONDON', 'BEIJING']
    organization_pool = ['SCHOOL1', 'SCHOOL2', 'SCHOOL3', 'SCHOOL4', 'SCHOOL5']
    event_pool = ['MEETING1', 'MEETING2', 'MEETING3', 'MEETING4', 'MEETING5']
    work_of_art_pool = ['MONALISA', 'The Pied Piper of Hamelin']
    consumer_good_pool = ['MEETING1', 'MEETING2', 'MEETING3', 'MEETING4', 'MEETING5']
    other_pool = ['SOMETHING1', 'SOMETHING2', 'SOMETHING3', 'SOMETHING4', 'SOMETHING5']

    google_entity_type ={0:'UNKNOWN', 1:'PERSON', 2:'LOCATION', 3:'ORGANIZATION', 4:'EVENT', 5:'WORK_OF_ART', 6:'CONSUMER_GOOD', 7:'OTHER'}
    entity_type ={0:'X', 1:iter(person_pool), 2:iter(location_pool), 3:iter(organization_pool), 4:iter(event_pool), 5:iter(work_of_art_pool), 6:iter(consumer_good_pool), 7:iter(other_pool)}
    entity_type2 ={0:'X', 1:iter(person_pool), 2:iter(location_pool), 3:iter(organization_pool), 4:iter(event_pool), 5:iter(work_of_art_pool), 6:iter(consumer_good_pool), 7:iter(other_pool)}
    #entity_type ={0:'X', 1:'DAVID', 2:'WASHINTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}

    setting_time = time.time()

    Times = ['the day after tomorrow', 'tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    line_time = ''
    for i in Times:
        if i in x_text:
            line_time = i
            x_text = x_text.replace(i, 'TIME1')

    found_time = time.time()

    client = language.LanguageServiceClient()
    document = types.Document(
            content=x_text,
            language='en',
            type=enums.Document.Type.PLAIN_TEXT)

    entities = client.analyze_entities(document).entities
    #print('Entities: {}'.format(entities))

    result = x_text 

    for entity in entities:
        result = result.replace(entity.name, next(entity_type[entity.type]))

    entity_time = time.time()
    #print("Replace nouns: {}".format(result))

    if light_module:
        values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, result)
        #embedding_init_time = time.time()
        test_message_embeddings = session.run(
                encodings,
                feed_dict={input_placeholder.values: values,
                    input_placeholder.indices: indices,
                    input_placeholder.dense_shape: dense_shape})
    else:
        #embedding_init_time = time.time()
        print("result %s" % result)
        test_message_embeddings = session.run(embedding, feed_dict={messages: [result]})
        #test_message_embeddings = session.run(embedding, feed_dict={messages: [result]})
        #embedding = embed([result])
        #test_message_embeddings = session.run(embedding)

    embedding_time = time.time()

    test_labels = [1]

    minimum = 100
    minimum_index = 0
    for i, message_embedding in enumerate(message_embeddings):
        error = rmse(np.array(message_embedding), np.array(test_message_embeddings))
        if minimum > error:
            minimum = error
            minimum_index = i

    # print("Minimum RMSE value: {}".format(minimum))
    # print("Estimation: {}".format(minimum_index))
    # print("Most similar script: {}".format(scripts[minimum_index]))
    #print("Answer: {}\n".format(test_label))
    result2 = scripts[minimum_index] #query


    Dict_entitis={}
    K=[] #Keys = Type
    V=[] #Values = Names
    google=[]
    for entity in entities:
        # Dict_entitis[entity.name]=entity_type[entity.type]
        google.append(google_entity_type[entity.type])
        #K.append(entity_type[entity.type])
        K.append(next(entity_type2[entity.type]))
        V.append(entity.name)


    # print("Dict_entitis : ", Dict_entitis)
    # print("Entities : ", google)
    print("Keys : ", K)
    print("Values : ", V)

    # for key, value in Dict_entitis.items():
    #     result2 = result2.replace(value, key)

    number = len(K)
    for i in range(0, number):
        result2 = result2.replace(K[i], V[i], 1)
    if 'TIME1' in result2:
        result2 = result2.replace('TIME1', line_time)

    end_time = time.time()

    print("input: {}".format(x_text))
    print("Query: {}".format(result2))
    print("setting_time={}".format(setting_time-start_time))
    print("entity_time={}".format(entity_time-setting_time))
    #print("make_bin_time={}".format(make_bin_time-entity_time))
    #print("embedding_init_time={}".format(embedding_init_time-entity_time))
    print("embedding_time={}".format(embedding_time-entity_time))
    print("rmse_time={}".format(end_time-embedding_time))
    print("total_time={}".format(end_time-start_time))
    
    if len(lines) > 1:
        y[test_enum] = y[test_enum].upper()
        y[test_enum] = y[test_enum].replace(" ", "")
        result2 = result2.upper()
        result2 = result2.replace(" ", "")
        #if int(y[test_enum]) == int(minimum_index):
        #print("y_estimate:{}, y:{}".format(minimum_index, y[test_enum]))
        print("y_estimate:{}, y:{}".format(result2, y[test_enum]))
        if y[test_enum] == result2:
            correct = 'O'
            answer_correct = answer_correct + 1
            print('answer_correct')
        else:
            correct = 'X'

    print("{}/{}".format(answer_correct, test_enum + 1))
    print("////////////////////////")

    #('enum, input, answer, estimate, correct')
    result_sheet_file = open('./result_sheet.txt', 'a')
    result_sheet_file.write("{}||{}||{}||{}||{}\n".format(test_enum, x_text, y[test_enum], result2, correct))
    result_sheet_file.close()
    start_time = time.time()

if len(lines) > 1:
    print("Result: {}/{}".format(answer_correct, len(lines)))
result_sheet_file.close()
