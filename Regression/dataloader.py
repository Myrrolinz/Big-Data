import os
import numpy as np
from Regression.utils import time_test

SKIP_NOTHI = 0
SKIP_BREAK = 1
SKIP_CONTI = 2

class DataLoader:

    def __init__(self, root, use_attr=False, num_attr=2, norm=True, lower=-1, upper=-1, fine_tune_ids=None,
                 add_test=True) -> None:
        self.user_voc = {}  # userName -> userID
        self.item_voc = {}  # itemName -> itemID
        self.user_voc_r = {}  # itemID -> itemName
        self.item_voc_r = {}  # userID -> userName
        self.user2item = []  # userID -> (itemID, score) list
        self.user2item_test = {}  # userID -> (itemID, score) list
        self.user2item_test_name = {}  # userID -> itemName in test file
        self.user2item_norm = []  # userID -> (score_mean, score_std)
        # self.item2user = [] # itemID -> (userID, score) list
        # self.item2user_norm = []  # itemID -> (score_mean, score_std)
        self.item2attr = []  # itemID -> item Attribute list
        # self.train_test_split = [] # userid -> itemID list split num
        self.trainfile = os.path.join(root, "train.txt")
        self.testfile = os.path.join(root, "test.txt")
        self.attribute_file = os.path.join(root, "itemAttribute.txt")
        self.use_attr = use_attr
        self.num_attr = num_attr
        self.norm = norm
        self.fine_tune_ids = None
        self.lower = lower
        self.upper = upper
        self.use_bounder = False
        if fine_tune_ids is not None:
            self.fine_tune_ids = set(fine_tune_ids)
            print("Will load fine tune ids")
        elif 0 <= lower < upper:
            self.use_bounder = True
            print(f"Will load range[{lower}, {upper})")
        else:
            print("Will load all")
        self.add_test = add_test
        # read relationship
        time_test("read from train file", self.read_train)
        print(f"N = {self.get_num_items()}, U = {self.get_num_users()} after load train data")
        if self.norm:
            time_test("normalize user2item", self.nomalize, self.user2item, self.user2item_norm)
            # time_test("normalize item2user", self.nomalize, self.item2user, self.item2user_norm)
        if self.use_attr:
            time_test("sort user2item", self.sort_user2items)
            time_test("read from attribute file", self.read_attribute)
            print(f"N = {self.get_num_items()} after read attribute file")
        if add_test:
            time_test("read from test file", self.read_test)
            print(f"N = {len(self.user2item_test)} in test data")

    # def load_data(self, batch_size=16):
    #     """[Deprecate method]load a batch of datas once, is a generator.
    #     Each batch itemID is in [pos, min(pos+B, N)]
    #
    #     Returns:
    #         x: items' feature, B*D
    #         y: items' true score, B*U
    #         r: items' is scored, B*U
    #     """
    #     N = self.get_num_items()
    #     D = self.get_num_attrs()
    #     U = self.get_num_users()
    #     for pos in range(0, N, batch_size):
    #         x = np.zeros((batch_size, D), dtype=np.float32)
    #         for i in range(pos, min(N, pos + batch_size)):
    #             attr = self.item2attr[i]
    #             if len(attr) == 2 and attr[0] is not None and attr[1] is not None:
    #                 x[i - pos] = attr
    #         y = np.zeros((batch_size, U), dtype=np.float32)
    #         r = np.zeros((batch_size, U), dtype=np.int8)
    #         for u in range(U):
    #             item_ids, scores = zip(*self.user2item[u])
    #             item_ids = np.array(item_ids)
    #             scores = np.array(scores)
    #             mask = (pos <= item_ids) * (item_ids < min(N, pos + batch_size))
    #             batch_idx = np.where(mask)[0]  # the index of self.user2item[u] that the itemID between pos and pos + B
    #             batch_idx2 = item_ids[batch_idx] - pos
    #             r[batch_idx2, u] = 1
    #             y[batch_idx2, u] = scores[batch_idx]
    #         yield x, y, r

    def sort_user2items(self):
        """sort each user's items list by itemID
        """
        for i, items in enumerate(self.user2item):
            self.user2item[i] = sorted(items, key=lambda x: x[0])

    def get_num_users(self):
        return len(self.user_voc)

    def get_num_items(self):
        return len(self.item_voc)

    def get_num_attrs(self):
        return self.num_attr

    def add_user(self, user_name):
        user_id = self.user_voc.get(user_name)
        if user_id is None:
            user_id = len(self.user2item)
            self.user_voc[user_name] = user_id
            self.user2item.append([])
            self.user_voc_r[user_id] = user_name
        return user_id

    def add_item(self, item_name):
        item_id = self.item_voc.get(item_name)
        if item_id is None:
            item_id = len(self.item_voc)
            self.item_voc[item_name] = item_id
            # self.item2user.append([])
            if self.use_attr:
                self.item2attr.append([])
            self.item_voc_r[item_id] = item_name
        return item_id

    def skip(self, user_id):
        if self.fine_tune_ids is not None and user_id not in self.fine_tune_ids:
            return SKIP_CONTI
        if self.use_bounder:
            if user_id < self.lower:
                return SKIP_CONTI
            elif user_id >= self.upper:
                return SKIP_BREAK
        return SKIP_NOTHI

    def get_user2item_userids(self):
        if self.fine_tune_ids is not None:
            return self.fine_tune_ids
        if self.lower < self.upper:
            return list(range(self.lower, self.upper))
        return None  # means all

    def read_train(self):
        scan_users = -1
        with open(self.trainfile, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                lst = line.strip().split('|')
                if len(lst) == 2:
                    scan_users += 1
                    action = self.skip(scan_users)
                    if action == SKIP_CONTI:
                        continue
                    elif action == SKIP_BREAK:
                        break
                    user_id = self.add_user(lst[0])
                    user_nitem = int(lst[1])
                    for j in range(user_nitem):
                        line = fp.readline().strip()
                        lst2 = line.split()
                        item_id = self.add_item(lst2[0])
                        score = int(lst2[1])
                        self.user2item[user_id].append((item_id, score))
                        # self.item2user[item_id].append((user_id, score))

    @staticmethod
    def dump_test_uid(ids):
        with open("Data/test_uid.txt", "w") as fp:
            for id in ids:
                fp.write(f"{id}\n")

    @staticmethod
    def load_test_uid(path):
        rtn = None
        if path is not None and len(path) > 0 and os.path.exists(path):
            rtn = []
            with open(path, "r") as fp:
                for line in fp:
                    rtn.append(int(line.strip()))
        return rtn

    def read_test(self):
        with open(self.testfile, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                lst = line.strip().split('|')
                if len(lst) == 2:
                    user_id = self.user_voc.get(lst[0], None)
                    if user_id is None:
                        print(f"No user {lst[0]} in train file, ignore it!")
                        continue
                    self.user2item_test[user_id] = []
                    user_nitem = int(lst[1])
                    for j in range(user_nitem):
                        line = fp.readline()
                        lst2 = line.strip()
                        item_id = self.item_voc.get(lst2, None)
                        if item_id is None:
                            print(f"No item {lst2} in train file!")
                            if user_id not in self.user2item_test_name.keys():
                                self.user2item_test_name[user_id] = []
                            self.user2item_test_name[user_id].append(lst2)
                        self.user2item_test[user_id].append((item_id, None))
        # DataLoader.dump_test_uid(self.user2item_test)

    def read_attribute(self):
        # TODO: Remain the items that have enough attributes
        with open(self.attribute_file) as fp:
            for line in fp:
                lst = line.strip().split('|')
                item_id = self.add_item(lst[0])
                self.item2attr[item_id] = list(map(eval, lst[1:]))

    def nomalize(self, table, output):
        for i, tups in enumerate(table):
            ids, scores = zip(*tups)
            s_mean = 0
            s_std = 100
            output.append((s_mean, s_std))
            scores = (np.array(scores) - s_mean) / s_std
            table[i] = list(zip(ids, scores.tolist()))

    def retransform(self, scores, input):
        scores2 = []
        if not self.norm:
            return scores
        assert (len(scores) == len(input))
        for i, slist in enumerate(scores):
            s_mean, s_std = input[i]
            scores2.append((np.array(slist) * s_std + s_mean).tolist())
        return scores2

    def dump_results(self, scores, result_path):
        user_avg_score = [0] * self.get_num_users()
        for u, items in enumerate(self.user2item):
            _, y = zip(*items)
            ascore = (sum(y) / len(y))
            if self.norm:
                ascore = ascore * self.user2item_norm[u][1] + self.user2item_norm[u][0]
            user_avg_score[u] = ascore
        assert(len(scores) == len(self.user2item_test))
        with open(result_path, "w") as fp:
            for j, (uid, items) in enumerate(self.user2item_test.items()):
                uname = self.user_voc_r[uid]
                fp.write(f"{uname}|{len(items)}\n")
                for i, (iid, _) in enumerate(items):
                    if iid is not None:
                        iname = self.item_voc_r[iid]
                        score = scores[j][i]
                    else:
                        iname = self.user2item_test_name[uid][0]
                        self.user2item_test_name[uid] = self.user2item_test_name[uid][1:]
                        score = user_avg_score[uid]
                    score = int(round(score))
                    if score < 0:
                        score = 0
                    if score > 100:
                        score = 100
                    fp.write(f"{iname} {score}\n")
