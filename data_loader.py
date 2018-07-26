import pprint

def load_data(file_name):
    data =  [line.rstrip() for line in list(open(file_name, "r").readlines())]
    x_text = []
    y_script = []
    y_category = []

    for i, sentence in enumerate(data):
        split_result = sentence.split("||")
        x_text.append(split_result[1])
        y_script.append(split_result[2])
        y_category.append(split_result[3])

    return [x_text, y_script, y_category] 

if __name__ == '__main__':
    pprint.pprint(load_data('./dataset/test.txt'))
