import time

import numpy as np
from Regression.dataloader import DataLoader
from Regression.config import CKPT_PATH
from Regression.utils import save_ckpt, load_ckpt
from Evaluation.eval import evaluate2
import os
import sys
import random


class CFLR:
    NAME = "CFLR"

    @staticmethod
    def get_ckpt_path(ckpt):
        dir = os.path.join(CKPT_PATH, CFLR.NAME)
        os.makedirs(dir, exist_ok=True)
        return os.path.join(dir, f"{CFLR.NAME}_{ckpt}.pth")


    def __init__(self, num_users, num_items, num_features, init_std):
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.weight = np.random.randn(num_users, num_features) * init_std
        self.x = np.random.randn(num_items, num_features) * init_std
        # self.lamb = None  # depreciate

    def backward_one_batch(self, beg, end, shuffle_idx, user_ids, user2items, user2scores, lr, lamb):
        batch_items = 0  # number of items in this batch
        batch_idx_list = list(range(beg, end)) if shuffle_idx is None else shuffle_idx[beg:end]
        for b in batch_idx_list:
            uid = b if user_ids is None else user_ids[b]
            items = user2items[b]
            batch_items += len(items)
            u_weight = self.weight[uid]
            dw = np.zeros_like(u_weight)
            for i, (iid, y) in enumerate(items):
                diff = user2scores[b][i] - y
                dw += diff * self.x[iid]
                dx = diff * u_weight + lamb * self.x[iid]
                self.x[iid] -= lr * dx
            dw += lamb * u_weight
            self.weight[uid] -= lr * dw
        return batch_items

    def train_one_epoch(self, user2items, lr, lamb, print_freq, user_ids, batch_size, shuffle):
        shuffle_idx = None
        if user_ids is not None:
            assert(len(user_ids) == len(user2items)), f"{len(user_ids)} != {len(user2items)}"
        if shuffle:
            shuffle_idx = list(range(len(user2items)))
            random.shuffle(shuffle_idx)
        assert(batch_size >= 1)

        user2scores = [[] for _ in range(len(user2items))]  # userID (relative pos) -> predict scores list, ordered by itemID in user2items
        loss = 0.0
        total_items = 0
        no_batch = 0  # batch
        num_batch = (len(user2items) + batch_size - 1) // batch_size # number of batchs
        bloss = 0.0  # loss of a batch
        for _j in range(len(user2items)):
            j = shuffle_idx[_j] if shuffle else _j
            items = user2items[j]
            if (len(items) == 0):
                continue
            uid = j if user_ids is None else user_ids[j]
            u_weight = self.weight[uid]
            total_items += len(items)
            for i, (iid, y) in enumerate(items):
                y_hat = u_weight @ self.x[iid].T
                user2scores[j].append(y_hat)
                bloss += ((y_hat - y) ** 2 + lamb * np.sum(self.x[iid]**2))
            bloss += lamb * np.sum(u_weight ** 2)
            if np.isnan(bloss):
                print("Loss is nan!")
                sys.exit(1)
            loss += bloss
            if (_j + 1) % batch_size == 0:
                batch_items = self.backward_one_batch(no_batch * batch_size,
                                                      (no_batch + 1) * batch_size,
                                                      shuffle_idx,
                                                      user_ids,
                                                      user2items,
                                                      user2scores,
                                                      lr,
                                                      lamb)
                no_batch += 1
                if no_batch % print_freq == 0:
                    print(f"({no_batch}/{num_batch}), lr {lr}, lambda {lamb}, bitems {batch_items}, avg loss {bloss / batch_items:.4f}")
                bloss = 0.0

        if no_batch < num_batch:
            batch_items = self.backward_one_batch(no_batch * batch_size,
                                                  len(user2items),
                                                  shuffle_idx,
                                                  user_ids,
                                                  user2items,
                                                  user2scores,
                                                  lr,
                                                  lamb)
            no_batch += 1
            print(f"({no_batch}/{num_batch}), lr {lr}, bitems {batch_items}, avg loss {bloss / batch_items:.4f}")
        loss /= total_items
        return loss

    def predict(self, user2items, user_ids):
        if user_ids is not None:
            assert(len(user_ids) == len(user2items))
        user2scores = [[] for _ in
                       range(len(user2items))]  # userID(relative position) -> predict scores list, ordered by itemID in user2items
        for j, items in enumerate(user2items):
            if (len(items) == 0):
                continue
            uid = j if user_ids is None else user_ids[j]
            u_weight = self.weight[uid]
            for iid, y in items:
                if iid is not None:
                    y_hat = u_weight @ self.x[iid].T
                    user2scores[uid].append(y_hat)
                else:
                    user2scores[uid].append(0)
        return user2scores



def eval2(model:CFLR, dataloader:DataLoader, *args, **kwargs):
    print("Begin Evaluating")
    y = [[item[1] for item in items] for items in dataloader.user2item]
    y_hat = model.predict(dataloader.user2item, dataloader.get_user2item_userids())
    y = dataloader.retransform(y, dataloader.user2item_norm)
    y_hat = dataloader.retransform(y_hat, dataloader.user2item_norm)
    evals = evaluate2(y_hat, y, *args, **kwargs)
    print(evals)
    print("Finish Evaluating")


def test2(model:CFLR, dataloader:DataLoader, result_path):
    print("Begin Test")
    assert(dataloader.add_test)
    y_hat = model.predict(dataloader.user2item_test.values(), list(dataloader.user2item_test.keys()))
    y_hat = dataloader.retransform(y_hat, dataloader.user2item_norm)
    dataloader.dump_results(y_hat, result_path)
    print("Finish Test")


def train2(model:CFLR, dataloader:DataLoader, nepoch, lr, lamb, start_epoch, print_freq, batch_size, shuffle):
    print("Begin Training")
    last_epoch = time.time()
    y = [[item[1] for item in items] for items in dataloader.user2item]# scores
    y_ori = dataloader.retransform(y, dataloader.user2item_norm)
    for epoch in range(start_epoch + 1, nepoch + 1):
        print(f"=========EPOCH[{epoch}/{nepoch}]==========")
        loss = model.train_one_epoch(dataloader.user2item, lr, lamb=lamb, print_freq=print_freq,
                                     user_ids=dataloader.get_user2item_userids(),
                                     batch_size=batch_size, shuffle=shuffle)
        save_ckpt(CFLR.get_ckpt_path(epoch), model)
        y_hat = model.predict(dataloader.user2item, dataloader.get_user2item_userids())
        y_hat = dataloader.retransform(y_hat, dataloader.user2item_norm)
        evals = evaluate2(y_hat, y_ori)
        print(f"[{epoch}/{nepoch}] avg loss = {loss:.4f}, time = {time.time() - last_epoch:.2f}s")
        print(evals)
        # do_log(self.ckpt_path, loss, train_evals, test_evals, epoch, nepoch, self.name)
        last_epoch = time.time()
    print("Finish Training")


def compare2(rpath1, rpath2, *args, **kwargs):
    print("Begin Compare")
    def read_result(rpath):
        user2scores = []
        with open(rpath) as fp:
            while(True):
                line = fp.readline()
                if not line:
                    break
                lst = line.strip().split('|')
                if len(lst) == 2:
                    # u = int(lst[0])
                    user2scores.append([])
                    nitem = int(lst[1])
                    for i in range(nitem):
                        line = fp.readline().strip()
                        score = int(line.split()[1])
                        user2scores[-1].append(score)
        return user2scores
    user2scores1 = read_result(rpath1)
    user2scores2 = read_result(rpath2)
    evals = evaluate2(user2scores1, user2scores2, *args, **kwargs)
    print(evals)
    print("Finsh Compare")


# ==================================================

class ContentLinearModel:

    def __init__(self, data_loader: DataLoader, lamb) -> None:
        self.data_loader = data_loader
        self.num_users = data_loader.get_num_users()          # U
        self.num_items = data_loader.get_num_items()          # N
        self.num_features = data_loader.get_num_attrs()    # D
        self.weight0 = np.random.random((self.num_users, self.num_features)) * 0.01  # U * D
        self.bias0 = np.zeros((self.num_users, )) # U
        self.lamb = lamb
        self.ckpt_path = os.path.join(CKPT_PATH, "ContentLinear")
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.name = "ContentLinearModel"

    def forward(self, x, mask):
        B = x.shape[0]
        y_hat = np.zeros((B, self.num_users))
        for i in range(B):
                for j in range(self.num_users):
                    if mask[i][j]:
                        y_hat[i][j] = (self.weight0[j].T @ x[i] + self.bias0[j])
        return y_hat
    
    def criterion(self, y_hat, y):
        return 0.5 * (np.sum((y_hat - y)**2) + self.lamb * np.sum(self.weight0**2))

    def backward(self, x, y_hat, y, mask, lr):
        B = x.shape[0]
        for j in range(self.num_users):
            dw = np.zeros((self.num_features, ))
            db = 0.0
            for i in range(B):
                if mask[i][j]:
                    dw += ((y_hat[i][j] - y[i][j]) * x[i] + self.lamb * self.weight0[j])
                    db += ((y_hat[i][j] - y[i][j]))
            self.weight0[j] -= lr * dw
            self.bias0[j] -= lr * db


def train(model:ContentLinearModel, nepoch, lr, start_epoch=0, print_freq=1):
    print("Begin Training")
    if start_epoch > 0:
        ckpt_name = os.path.join(model.ckpt_path, f"{model.name}_{start_epoch}.pth")
        if os.path.exists(ckpt_name):
            model = load_ckpt(ckpt_name)

    last_epoch = time.time()
    # x_test, y_test, r_test = self.data_loader.load_data(train=False)
    for epoch in range(1, nepoch + 1):
        sum_loss = 0
        sum_n = 0
        last_iter = time.time()
        for i, data in enumerate(model.data_loader.load_data()):
            x, y, r = data
            sum_n += x.shape[0]
            y_hat = model.forward(x, r)
            loss = model.criterion(y_hat, y)
            if np.isnan(loss):
                print(f"loss is Nan! Training exit")
                sys.exit(1)
            sum_loss += loss
            model.backward(x, y_hat, y, r, lr)
            if (i + 1) % print_freq == 0:
                print(f"Iter({i}), loss {sum_loss / sum_n:.4f} {time.time() - last_iter: .2f}s")
                last_iter = time.time()
            # train_evals = evaluate(y_hat, y, r, save_ori=False)
        print(f"=========EPOCH[{epoch}/{nepoch}]==========")
        print(f"[{epoch}/{nepoch}] loss = {sum_loss / sum_n:.4f} {time.time() - last_epoch:.2f}s")
        # print(train_evals)
        # print(test_evals)
        # do_log(self.ckpt_path, loss, train_evals, test_evals, epoch, nepoch, self.name)
        ckpt_name = os.path.join(model.ckpt_path, f"{model.name}_{epoch}.pth")
        save_ckpt(ckpt_name, model)
        last_epoch = time.time()
    print("Finish Training")
    