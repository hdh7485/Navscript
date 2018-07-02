def load_data(file_name):
    data =  [line.rstrip() for line in list(open(file_name, "r").readlines())]
    x_text = []
    y = []
    for i, x in enumerate(data):
        if i % 2 == 0:
            x_text.append(x)
        else:
            y.append(x)

    return [x_text, y] 

if __name__ == '__main__':
    print(load_data('./dataset/test.txt'))
