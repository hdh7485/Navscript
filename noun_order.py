def check_order(sentence, nouns):
    position = {}
    for noun in nouns:
        position[noun] = sentence.find(noun)
    result = sorted(position, key=position.__getitem__)
    return result

def main():
    sentences_set = ['Where is a burgerking along the route.', 'What\'s the traffic like on my route?', 'Can you find me a gas station with restroom facilities nearby?']
    nouns_set = [['route', 'burgerking', 'along', 'Where'], ['route', 'traffic'], ['facilities', 'restroom', 'gas station']]

    for sentence, nouns in zip(sentences_set, nouns_set):
        print('input:{}'.format(sentence))
        print('input:{}'.format(nouns))
        print('output:{}'.format(check_order(sentence, nouns)))

if __name__ == '__main__':
    main()
