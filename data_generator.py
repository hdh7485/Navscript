def make_data():
    place = ['burger king', 'burgerking', 'restroom', 'restroom facility',  'super market', 'supermarket', 'school', 'starbucks', 'coffee shop', 'cafe', 'college', 'gas station', 'park', 'department store', 'restaurent', 'indoor parking', 'outdoor parking', 'my destination']
    other = ['route', 'traffic camera', 'traffic', 'speed camera']
    weather = ['weather', 'weather forecast']
    geocode = ['US-101', 'Bayshore Blvd', 'I-580 East']
    event = ['meeting', 'party', 'speech']
    time = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    distance = ['50 meters', '100 meters', '500 meters', '1000 meters', '2000 meters']
    service = ['valet service', 'credit card']
    search_place = ['San Francisco Museum of Modern Art', 'Downtown Berkeley', 'Bay Bridge', 'Oakland', 'Seoul', 'New York', 'Wahinton', 'Chicago']

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

    result = []
    for sentence in sentences:
        for p in place:
            result.append(sentence.replace("[place]", p))
    print(result)
    with open('generate_test.txt', 'w') as f:
        for i in result:
            f.write("{}\n".format(i))
    return

if __name__ == "__main__":
    make_data()
