from config import *
from utils import *
from cf_class import *
from user import *

if __name__ == '__main__':
    print(Alg_information)

    if_pre = int(input("如果重新训练模型 输入1 加载现有模型 输入0："), 10)
    if if_pre == 1:
        c = CF_user(Data_train, Data_test)
        c.build(c.train_p)
        save_class(c, Save_path, user_class_path)
        c.static_analyse()
        c.train()
        save_class(c, Save_path, user_class_path+'.trained')
        c.test(c.test_p)
        save_class(c, Save_path, user_class_path+'.tested')
    else:
        c = load_class(Save_path, user_class_path+'.trained')
        if not c.if_build:
            c.build(Data_train)
            c.train()
            save_class(c, Save_path, user_class_path)
        elif not c.if_train:
            c.train()
            save_class(c, Save_path, user_class_path+'.trained')
        elif not c.if_test:
            c.test(Data_test)
            save_class(c, Save_path, user_class_path + '.tested')
    c = load_class(Save_path, user_class_path+'.trained')
    c.test(Data_test)
    save_class(c, Save_path, user_class_path + '.tested')
    c = load_class(Save_path, user_class_path+'.tested')
    for i in range(len(c.r)):
        for j in range(len(c.r[i])):
            if c.r[i][j][1] < 0:
                c.r[i][j] = (c.r[i][j][0], int(abs(c.r[i][j][1])))
            if c.r[i][j][1] > 100:
                c.r[i][j] = (c.r[i][j][0], 100)
            else:
                c.r[i][j] = (c.r[i][j][0], int(c.r[i][j][1]))

    with open(result_path, 'w', encoding='utf-8') as f:
        for i in range(len(c.r)):
            f.write(str(i) + '|6\n')
            for j in range(len(c.r[i])):
                f.write(str(c.r[i][j][0])+' '+str(c.r[i][j][1])+'\n')
    save_class(c, Save_path, user_class_path + '.tested.re')

