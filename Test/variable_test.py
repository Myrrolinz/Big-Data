import random

class SomeClass:
    def __init__(self):
        self.alist = [1, 2, 3, 4]
        self.blist = [[1, 2], [2, 3], [3, 4], [4, 5, 6]]

    def get_alist(self):
        return self.alist

def some_fun(a, b):
    a += 1
    b = 0


if __name__ == '__main__':
    sc = SomeClass()
    # lst = sc.get_alist()
    # random.seed(20)
    # random.shuffle(lst)
    # print(sc.alist)
    # random.shuffle(sc.alist)
    # print(sc.alist)

    # lst = sc.blist.copy()
    # random.shuffle(lst)
    # print(sc.blist)
    # lst[0] = [0, 1, 2]
    # print(sc.blist)
    # tmp = list(zip(sc.alist, sc.blist))
    # random.shuffle(tmp)
    # lst1, lst2 = zip(*tmp)
    # print(lst1, lst2)

    a = 1
    b = 1
    some_fun(a, b)
    print(a, b)

