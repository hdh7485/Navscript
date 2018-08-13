def check_order(sentence, nouns):
    position = {}
    for noun in nouns:
        position[noun] = sentence.find(noun)
    result = sorted(position, key=position.__getitem__)
    return result

def main():
    sentence = 'Where is a burgerking along the route.'
    #nouns = ['burgerking', 'route']
    nouns = ['route', 'burgerking', 'along', 'Where']
    print(sentence)
    print(check_order(sentence, nouns))

if __name__ == '__main__':
    main()
