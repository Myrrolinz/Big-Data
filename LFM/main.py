from LFM import LFM
import config
import os


def train():
    lfm_model = LFM(iter_num=10)
    lfm_model.load_data()
    lfm_model.init_latent_matrix()
    lfm_model.train()


def get_test():
    lfm_model = LFM()
    lfm_model.init_latent_matrix()
    lfm_model.get_test_result(test_flag=True)


def eval():
    ckpt_files = os.listdir(config.CKPT_FILE_FOLDER)
    epoch_list = []
    for i in ckpt_files:
        epoch = int(i.split("_")[0])
        epoch_list.append(epoch)
    epoch_list = list(set(epoch_list))
    print("epoch list: ", epoch_list)

    lfm_model = LFM()
    lfm_model.load_data()
    for epoch in epoch_list:
        lfm_model.calc_train(epoch)


if __name__ == "__main__":
    train()
    get_test()
