import os

from Regression.modelzoo import CFLR, train2, eval2, test2
from Regression.dataloader import DataLoader
from Regression.config import DATA_PATH
from Regression.utils import load_ckpt, setup_seed, time_test
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Script of Collaborative Filtering Linear Regression Model")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "test"], required=True,
                        help="The mode of this script")
    parser.add_argument("--ckpt", type=int, default=0, help="Load from checkpoint if ckpt > 0")
    # About DataLoader
    parser.add_argument("--norm", type=bool, default=True, help="Whether to nomalize the scores")
    # parser.add_argument("--lower", type=int, default=-1, help="User id in [lower, upper) will be loaded, used to debug")
    parser.add_argument("--upper", type=int, default=-1, help="User id in [lower, upper) will be loaded, used to debug")
    # parser.add_argument("--ftune", type=str, default="", help="Choose username by file to fine tune the model")
    # About Model
    parser.add_argument("--numf", type=int, default=6, help="Number of features of CFLR")
    parser.add_argument("--init", type=float, default=0.5, help="Initialize the parameters' std")
    # About Training
    parser.add_argument("--lamb", type=float, default=0.01, help="Regularization coefficient when training")
    parser.add_argument("--nepoch", type=int, default=50, help="Training epoch")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate when training")
    parser.add_argument("--batch", type=int, default=1, help="Batch size when training")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle in each epoch")
    parser.add_argument("--pfreq", type=int, default=500, help="Print Iterations frequency")
    parser.add_argument("--seed", type=int, default=20, help="Random seed")
    # About Evaluating
    parser.add_argument("--rmse", action="store_true", help="Whether to calculate RMSE")
    parser.add_argument("--ptopk", action="store_true", help="Whether to calculate Precision@K")
    parser.add_argument("--rtopk", action="store_true", help="Whether to calculate Recall@K")
    parser.add_argument("--corr", action="store_true", help="Whether to calculate Spearman Rank Correlation")
    parser.add_argument("--sort", type=bool, default=True, help="See doc of EvalUtils.value2id")
    parser.add_argument("--thresh", type=int, default=60, help="See doc of EvalUtils.value2id")
    parser.add_argument("--k", type=int, default=10, help="See doc of EvalUtils.precision")

    args = parser.parse_args()
    print(args)

    setup_seed(args.seed)

    # uid = DataLoader.load_test_uid(args.ftune)

    data_loader = DataLoader(DATA_PATH,
                             use_attr=False,
                             norm=args.norm,
                             lower=0,
                             upper=args.upper,
                             fine_tune_ids=None,
                             add_test=(args.mode == "test"))

    ckpt_path = CFLR.get_ckpt_path(args.ckpt)
    if args.mode == "train":
        if args.ckpt > 0:
            assert os.path.exists(ckpt_path), f"No such file {ckpt_path}"
            model = load_ckpt(ckpt_path)
        else:
            model = CFLR(data_loader.get_num_users(),
                         data_loader.get_num_items(),
                         num_features=args.numf,
                         init_std=args.init)
        train2(model, data_loader, nepoch=args.nepoch,
               lr=args.lr, lamb=args.lamb, start_epoch=args.ckpt,
               print_freq=args.pfreq, batch_size=args.batch,
               shuffle=args.shuffle)

    elif args.mode == "eval":
        assert os.path.exists(ckpt_path), f"No such file {ckpt_path}"
        model = load_ckpt(ckpt_path)
        eval2(model, data_loader,
              rmse=args.rmse, ptopk=args.ptopk,
              rtopk=args.rtopk, corr=args.corr,
              k=args.k, sort=args.sort, thresh=args.thresh)
    else:
        assert os.path.exists(ckpt_path), f"No such file {ckpt_path}"
        model = load_ckpt(ckpt_path)
        result_path = f"result_{args.ckpt}.txt"
        time_test("dump result", test2, model, data_loader, result_path)

