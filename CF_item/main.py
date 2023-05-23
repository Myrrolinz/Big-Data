from config import *
from utils import *
from cf_class import *
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
    # model = load_class(Save_path, user_class_path+'.trained')
    # model.test(Data_test)
    # save_class(model, Save_path, user_class_path + '.tested')
    # model = load_class(Save_path, user_class_path+'.tested')
    for i in range(len(model.r)):
        for j in range(len(model.r[i])):
            if model.r[i][j][1] < 0:
                model.r[i][j] = (model.r[i][j][0], int(abs(model.r[i][j][1])))
            if model.r[i][j][1] > 100:
                model.r[i][j] = (model.r[i][j][0], 100)
            else:
                model.r[i][j] = (model.r[i][j][0], int(model.r[i][j][1]))

    with open(result_path, 'w', encoding='utf-8') as f:
        for i in range(len(model.r)):
            f.write(str(i) + '|6\n')
            for j in range(len(model.r[i])):
                f.write(str(model.r[i][j][0])+' '+str(model.r[i][j][1])+'\n')
    save_class(model, Save_path, user_class_path + '.tested.re')

