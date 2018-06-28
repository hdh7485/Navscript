import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#import re
import psutil
import humanize
import argparse
import sys

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import six

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

lines="What's the traffic like on my route?"
lines = sys.argv[1]

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

messages2 = ["What's the SOMETHING for this afternoon?",
        "What's the SOMETHING like on my SOMETHING?",
        "Show me a SOMETHING on WASHINGTON and WASHINGTON.",
        "Can you find me a WASHINGTON with WASHINGTON nearby?",
        "Find a WASHINGTON along WASHINGTON.",
        "Find the cheapest indoor SOMETHING within 500 meters of my SOMETHING.",
        "Okay, can you find me a WASHINGTON on my SOMETHING that has a WASHINGTON?",
        "Find SOMETHING near WASHINGTON that accepts SOMETHING and has a SOMETHING.",

        "Navigate to WASHINGTON.",
        "What's my SOMETHING to WASHINGTON?",
        "Show me alternative SOMETHING.",
        "Reroute using SOMETHING.",

        "Drive to WASHINGTON.",
        "What's my SOMETHING?",
        "Can I make tomorrow's 10am {EVENT:COMMON} without recharging?",
        "What's SOMETHING like on the WASHINGTON?",
        "Are there any SOMETHING on my SOMETHING?",
        "Will it rain tomorrow in WASHINGTON?"
        ]

scripts = ["[SEARCH FROM:SOMETHING WHERE:HERE WHEN:AFTERNOON]",
         "[SEARCH FROM:SOMETHING WHERE:SOMETHING]",
         "[SEARCH FROM:SOMETHING WHERE:[SEARCH GEOCODE WHERE:WASHINTON]]",
         "[SEARCH FROM:SCHOOL WHERE:NEARBY WITH:SOMETHING]",
         "[SEARCH ONE FROM:WASHINTON WHERE:WASHINTON]",
         "[SEARCH ONE FROM:SOMETHING WHERE:SOMETHING RANGE:500M WITH:[SORT PRICE ASC]]",
         "[SEARCH ONE FROM:WASHINTON WHERE:SOMETHING WITH:WASHINTON]",
         "[SEARCH ONE FROM:SOMETHING WITH:SOMETHING WITH:SOMETHING]",
         "[ROUTE TO:[SEARCH KEYWORD:WASHINTON]]",
         "[ROUTE INFO:SOMETHING]",
         "[ROUTE ALTROUTE]",
         "[ROUTE ALTROUTE USE:[SEARCH LINKS:SOMETHING]]",
         
         "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:WASHINTON]]]",
         "[MODE SOMETHING]",
         "[MODE SOMETHING TO:[SEARCH KEYWORD:MEETTING FROM:SCHEDULE WHEN:10AM] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
         "[MODE SOMETHING [SEARCH FROM:TRAFFIC WHERE:[SEARCH KEYWORD:WASHINTON]] WITH:[VOICERESPONSE TEMPLATE:""*]",
         "[MODE SOMETHING WHERE:SOMETHING WITH:[VOICERESPONSE TEMPLATE:""*]]",
         "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:WASHINTON] WHEN:TOMORROW]"
          ]


google_entity_type ={0:'UNKNOWN', 1:'PERSON', 2:'LOCATION', 3:'ORGANIZATION', 4:'EVENT', 5:'WORK_OF_ART', 6:'CONSUMER_GOOD', 7:'OTHER'}
entity_type ={0:'X', 1:'DAVID', 2:'WASHINTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
i_entity_type ={'X':0, 'DAVID':1, 'WASHINTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}

client = language.LanguageServiceClient()
document = types.Document(
        content=lines,
        language='en',
        type=enums.Document.Type.PLAIN_TEXT)

entities = client.analyze_entities(document).entities

result = lines

for entity in entities:
    result = result.replace(entity.name, entity_type[entity.type] )

print("Replace nouns: {}".format(result))


module_url = "https://tfhub.dev/google/universal-sentence-encoder/1" #@param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)


with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(messages2))

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  test_message_embeddings = session.run(embed([result]))

test_labels = [1]

minimum = 100
minimum_index = 0
for i, message_embedding in enumerate(message_embeddings):
    error = rmse(np.array(message_embedding), np.array(test_message_embeddings))
    if minimum > error:
      minimum = error
      minimum_index = i

print("\n\n\n\n\n")


print("Minimum RMSE value: {}".format(minimum))
print("Most similar script: {}".format(scripts[minimum_index]))
print("Estimation: {}".format(minimum_index))
#print("Answer: {}\n".format(test_label))
print()
result2 = scripts[minimum_index] #query

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

# for key, value in Dict_entitis.items():
#     result2 = result2.replace(value, key)

number = len(K)
for i in range(0, number):
    result2 = result2.replace(K[i], V[i],1)

print("input: {}".format(lines))
print("Query: {}".format(result2))
