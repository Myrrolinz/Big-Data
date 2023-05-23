import os
import pickle
import sys


def file_read(path):
    if os.path.exists(path):
        file = open(path, 'r', encoding='utf-8')
        return file
    else:
        print('ERROR: %s does not exit!' % (path,))


def file_write(path, if_a = 0):
    if os.path.exists(path) and if_a:
        file = open(path, 'a+', encoding='utf-8')
        return file
    else:
        file = open(path, 'w', encoding='utf-8')
        return file


def save_class(c, dir_path, file_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'wb') as file:
        pickle.dump(c, file)



def load_class(dir_path, file_path):
    if os.path.exists(dir_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                c = pickle.load(file)
            return c
        else:
            print("ERROR Save file %s does not exit!" % (file_path,))
    else:
        os.mkdir(dir_path)
        print("ERROR you sould save first!")


def swap(a, b):
    a = a^b
    b = a^b
    a = a^b
    return a,b

def devide(path, mid):
    full = load_class(path, 'sim.user')
    temp = []
    for i in range(mid):
        temp.append(full[i])
        full[i] = {}
    save_class(path, 'sim1.user')
    temp = []
    for i in range(mid, len(full)):
        temp.append(full[i])
        full[i] = {}
    save_class(path,'sim2.user')
