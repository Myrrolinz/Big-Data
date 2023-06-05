from config import *
from utils import *
from item import *
from user import *

if __name__ == '__main__':
    print(Alg_information)
    is_retrain = int(input("Please Choose:\t 1. Retrain the model\t 0. Load the model\nType 1 or 0: "))
    if is_retrain == 1:
        model = CF_user(Data_train, Data_test)
        model.build(model.train_p)
        save_class(model, Save_path, user_class_path)
        model.static_analyse()
        model.train()
        save_class(model, Save_path, user_class_path+'.trained')
        model.test(model.test_p)
        save_class(model, Save_path, user_class_path+'.tested')
    else:
        model = load_class(Save_path, user_class_path+'.trained')
        if not model.if_build:
            model.build(Data_train)
            model.train()
            save_class(model, Save_path, user_class_path)
        elif not model.if_train:
            model.train()
            save_class(model, Save_path, user_class_path+'.trained')
        elif not model.if_test:
            model.test(Data_test)
            save_class(model, Save_path, user_class_path + '.tested')


