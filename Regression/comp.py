import os
import argparse
from Regression.modelzoo import compare2
from Regression.utils import time_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The Script of Result Compare")
    # About Compare
    parser.add_argument("result1", type=str, help="Result File 1")
    parser.add_argument("result2", type=str, help="Result File 2")
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

    assert (os.path.exists(args.result1)), f"No such file {args.result1}"
    assert (os.path.exists(args.result2)), f"No such file {args.result2}"
    time_test("compare results", compare2,
              args.result1, args.result2,
              rmse=args.rmse, ptopk=args.ptopk,
              rtopk=args.rtopk, corr=args.corr,
              k=args.k, sort=args.sort, thresh=args.thresh)