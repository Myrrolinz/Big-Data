import os

Alg_information = 'Collaborative Filtering'
Data_folder = '../Data'
Save_path = './Save'
print_per = 7
save_per = 9
topn = 500
Thresh = 0
split_size = 0.1

Data_train = os.path.join(Data_folder, 'train.txt')
Data_itemattr = os.path.join(Data_folder, 'itemAttribute.txt')
Data_test = os.path.join(Data_folder, 'test.txt')
class_path = os.path.join(Save_path, 'class.pickle')

my_train = os.path.join(Data_folder, 'my_train.txt')
my_test = os.path.join(Data_folder, 'my_test.txt')
result_path = os.path.join(Save_path, 'result.txt')

user_class_path = os.path.join(Save_path, 'user.pickle')

if_pre = False
