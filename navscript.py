import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
import os
import pickle
import time
import data_loader

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

start_time = time.time()

if len(sys.argv) > 1:
    lines = [sys.argv[1]]
else:
    loaded_data = data_loader.load_data('./dataset/test.txt')
    lines = loaded_data[0]
    y = loaded_data[1]
    answer_correct = 0 

for test_enum, x_text in enumerate(lines):
    messages = ["What's the {OTHER:COMMON} for this afternoon?",
            "What's the {OTHER:COMMON} like on my {OTHER:COMMON}?",
            "Show me a {OTHER:COMMON} on {LOCATION:PROPER} and {LOCATION:PROPER}.",
            "Can you find me a {LOCATION:COMMON} with {LOCATION:COMMON} nearby?",
            "Find a {LOCATION:COMMON} along {LOCATION:COMMON}.",
            "Find the cheapest indoor {OTHER:COMMON} within 500 meters of my {OTHER:COMMON}.",
            "Okay, can you find me a {LOCATION:COMMON} on my {OTHER:COMMON} that has a {LOCATION:COMMON}?",
            "Find {OTHER:COMMON} near {LOCATION:COMMON} that accepts {OTHER:COMMON} and has a {OTHER:COMMON}.",

            "Navigate to {LOCATION:PROPER}.",
            "What's my {OTHER:PROPER} to {LOCATION:COMMON}?",
            "Show me alternative {OTHER:COMMON}.",
            "Reroute using {OTHER:PROPER}.",

            "Drive to {LOCATION:PROPER}.",
            "What's my {OTHER:COMMON}?",
            "Can I make tomorrow's 10am {EVENT:COMMON} without recharging?",
            "What's {OTHER:COMMON} like on the {LOCATION:PROPER}?",
            "Are there any {OTHER:COMMON} on my {OTHER:COMMON}?",
            "Will it rain tomorrow in {LOCATION:PROPER}?"
            ]

    messages2 = [line.rstrip('\n') for line in open('profile_messages.txt')]
    #print(messages2)

    scripts = ["[SEARCH FROM:SOMETHING1 WHERE:HERE WHEN:AFTERNOON]",
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
            "[MODE SOMETHING1 TO:[SEARCH KEYWORD:MEETTING FROM:SCHEDULE WHEN:10AM] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
            "[MODE SOMETHING1 [SEARCH FROM:TRAFFIC WHERE:[SEARCH KEYWORD:PLACE1]] WITH:[VOICERESPONSE TEMPLATE:""*]",
            "[MODE SOMETHING1 WHERE:SOMETHING2 WITH:[VOICERESPONSE TEMPLATE:""*]]",
            "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:PLACE1] WHEN:TOMORROW]"
            ]

    print('Input: {}'.format(x_text ))

    location_pool = ['PLACE1', 'PLACE2', 'PLACE3', 'PLACE4', 'PLACE5']
    #location_pool = ['WASHINGTON', 'SEOUL', 'MADRID', 'LONDON', 'BEIJING']
    other_pool = ['SOMETHING1', 'SOMETHING2', 'SOMETHING3', 'SOMETHING4', 'SOMETHING5']
    event_pool = ['MEETING1', 'MEETING2', 'MEETING3', 'MEETING4', 'MEETING5']

    google_entity_type ={0:'UNKNOWN', 1:'PERSON', 2:'LOCATION', 3:'ORGANIZATION', 4:'EVENT', 5:'WORK_OF_ART', 6:'CONSUMER_GOOD', 7:'OTHER'}
    entity_type ={0:'X', 1:'DAVID', 2:iter(location_pool), 3:'SCHOOL', 4:iter(event_pool), 5:'MONALISA', 6:'NOTEBOOK', 7:iter(other_pool)}
    entity_type2 ={0:'X', 1:'DAVID', 2:iter(location_pool), 3:'SCHOOL', 4:iter(event_pool), 5:'MONALISA', 6:'NOTEBOOK', 7:iter(other_pool)}
    #entity_type ={0:'X', 1:'DAVID', 2:'WASHINTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
    i_entity_type ={'X':0, 'DAVID':1, 'WASHINTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}

    setting_time = time.time()

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

    #print("Replace nouns: {}".format(result))

    #sentence-encoder/2
    #module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
    module_url = "/Users/hdh7485/navscript/modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"
    #sentence-encoder/1
    #module_url = "/Users/hdh7485/navscript/modules/c6f5954ffa065cdb2f2e604e740e8838bf21a2d3"
    #module_url =  "https://tfhub.dev/google/universal-sentence-encoder-lite/2"

    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    entity_time = time.time()

    if not os.path.exists("./profile.bin"):
        print("There is no profile.bin. Making profile.bin")
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(embed(messages2))
            with open('profile.bin', 'wb') as f:
                pickle.dump(message_embeddings, f)
        print("Finish make profile!")
    else:
        print("profile.bin exists")
        with open('./profile.bin', 'rb') as f:
            message_embeddings = pickle.load(f)

    make_bin_time = time.time()

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        test_message_embeddings = session.run(embed([result]))

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
    # print("Keys : ", K)
    # print("Values : ", V)

    # for key, value in Dict_entitis.items():
    #     result2 = result2.replace(value, key)

    number = len(K)
    for i in range(0, number):
        result2 = result2.replace(K[i], V[i],1)

    end_time = time.time()

    print("input: {}".format(x_text))
    print("Query: {}".format(result2))
    print("setting_time={}".format(setting_time-start_time))
    print("entity_time={}".format(entity_time-setting_time))
    print("make_bin_time={}".format(make_bin_time-entity_time))
    print("embedding_time={}".format(embedding_time-make_bin_time))
    print("rmse_time={}".format(end_time-embedding_time))
    
    if len(lines) > 1:
        if int(y[test_enum]) == int(minimum_index):
            answer_correct = answer_correct + 1
            print('answer_correct')
        print("y_estimate:{}, y:{}".format(minimum_index, y[test_enum]))
    print("////////////////////////")

if len(lines) > 1:
    print("Result: {}/{}".format(answer_correct, len(lines)))
