import os
import pickle
import sys


def file_read(path):
    if os.path.exists(path):
        file = open(path, 'r', encoding='utf-8')
        return file
    else:
        print('ERROR: %s does not exit!' % (path,))

def file_save(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f)
    
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

def rate_modify(rate):
    if rate < 0:
        return 0
    elif rate > 100:
        return 100
    else:
        return round(rate)

