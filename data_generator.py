def replace_tags(string_list):
    place = ['burger king', 'burgerking', 'restroom', 'restroom facility',  'super market', 'supermarket', 'school', 'starbucks', 'coffee shop', 'cafe', 'college', 'gas station', 'park', 'department store', 'restaurent', 'indoor parking', 'outdoor parking', 'my destination']
    other = ['route', 'traffic camera', 'traffic', 'speed camera']
    weather = ['weather', 'weather forecast']
    geocode = ['US-101', 'Bayshore Blvd', 'I-580 East']
    event = ['meeting', 'party', 'speech']
    time = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    distance = ['50 meters', '100 meters', '500 meters', '1000 meters', '2000 meters']
    service = ['valet service', 'credit card']
    search_place = ['San Francisco Museum of Modern Art', 'Downtown Berkeley', 'Bay Bridge', 'Oakland', 'Seoul', 'New York', 'Wahinton', 'Chicago']

    class_lists = []

    generated_sentences = []
    for sentence in string_list:
        if not sentence.find('[place]') == -1:
            for p in place:
                generated_sentences.append(sentence.replace("[place]", p))
        else:
            generated_sentences.append(sentence)

    generated_sentences2 = []
    for sentence in generated_sentences:
        if not sentence.find('[time]') == -1:
            for p in time:
                generated_sentences2.append(sentence.replace("[time]", p))
        else:
            generated_sentences2.append(sentence)

    generated_sentences3 = []
    for sentence in generated_sentences2:
        if not sentence.find('[weather]') == -1:
            for p in weather:
                generated_sentences3.append(sentence.replace("[weather]", p))
        else:
            generated_sentences3.append(sentence)

    generated_sentences4 = []
    for sentence in generated_sentences3:
        if not sentence.find('[other]') == -1:
            for p in other:
                generated_sentences4.append(sentence.replace("[other]", p))
        else:
            generated_sentences4.append(sentence)

    generated_sentences5 = []
    for sentence in generated_sentences4:
        if not sentence.find('[service]') == -1:
            for p in service:
                generated_sentences5.append(sentence.replace("[service]", p))
        else:
            generated_sentences5.append(sentence)

    generated_sentences6 = []
    for sentence in generated_sentences5:
        if not sentence.find('[geocode]') == -1:
            for p in geocode:
                generated_sentences5.append(sentence.replace("[geocode]", p))
        else:
            generated_sentences6.append(sentence)

    generated_sentences7 = []
    for sentence in generated_sentences6:
        if not sentence.find('[search_place]') == -1:
            for p in search_place:
                generated_sentences6.append(sentence.replace("[search_place]", p))
        else:
            generated_sentences7.append(sentence)

    generated_sentences8 = []
    for sentence in generated_sentences7:
        if not sentence.find('[distance]') == -1:
            for p in distance:
                generated_sentences7.append(sentence.replace("[distance]", p))
        else:
            generated_sentences8.append(sentence)

    replaced_result = generated_sentences8
    return replaced_result

def make_data():
    sentences = [
            "What's the [weather] for this [time]?",
            "What's the [other] like on my [other]?",
            "Show me a [other] on [geocode] and [geocode].",
            "Can you find me a [place] with [place] nearby?",
            "Find a [place] along route.",
            "Find the cheapest [place] within [distance] of [place].",
            "Okay, can you find me a [place] on my route that has a [place]?",
            "Find [place] near destination that accepts [service] and has a [service].",#Isn't it need range(near) query?

            "Navigate to [search_place].",
            "What's my ETA to destination? ",
            "Show me alternative routes.",
            "Reroute using [geocode].",

            "Drive to [search_place].",
            "What's my drive range?",
            "Can I make [time] [event] without recharging?",
            "What's traffic like on the [search_place]?",
            "Are there any [other] on my route?",
            "Will it rain [time] in [search_place]?"
            ]

    navscripts = [
            "[SEARCH FROM:[weather] WHERE:HERE WHEN:[time]]",
            "[SEARCH FROM:[other] WHERE:[other]]",
            "[SEARCH FROM:[ohter] WHERE:[SEARCH GEOCODE WHERE:[geocode] and [geocode]]]",
            "[SEARCH FROM:[place] WHERE:NEARBY WITH:[place]]",
            "[SEARCH ONE FROM:[place] WHERE:ALONGROUTE]",
            "[SEARCH ONE FROM:[place] WHERE:[place] RANGE:[distance] WITH:[SORT PRICE ASC]]",
            "[SEARCH ONE FROM:[place] WHERE:ONROUTE WITH:[place]]",
            "[SEARCH ONE FROM:[place] WITH:[service] WITH:[service]]",

            "[ROUTE TO:[SEARCH KEYWORD:[search_place]]]",
            "[ROUTE INFO:ETA]",
            "[ROUTE ALTROUTE]",
            "[ROUTE ALTROUTE USE:[SEARCH LINKS:[geocode]]]",

            "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:[search_place]]]]",
            "[MODE DRIVERANGE]",
            "[MODE DRIVERANGE TO:[SEARCH KEYWORD:[time] [event] FROM:SCHEDULE WHEN:[time]] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
            "[MODE TRAFFIC [SEARCH FROM:TRAFFIC WHERE:[SEARCH KEYWORD:[search_place]] WITH:[VOICERESPONSE TEMPLATE:""*]",
            "[MODE [other] WHERE:ONROUTE WITH:[VOICERESPONSE TEMPLATE:""*]]",
            "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:[place]] WHEN:[time]]"
            ]
    sentence_result = replace_tags(sentences)
    script_result = replace_tags(navscripts)
    print(sentence_result)
    print(script_result)

    with open('generate_test.txt', 'w') as f:
        for sentence, navscript in zip(sentence_result, script_result):
            f.write("{}\n{}\n".format(sentence, navscript))

    return

if __name__ == "__main__":
    make_data()
