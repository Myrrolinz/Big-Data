from Evaluation.eval import EvalUtils, evaluate
import numpy as np

def test_RMSE():
    a = [90, 80, 0, 100, 20, 0, 0]
    b = [70, 80, 100, 0, 30, 0, 0]
    arr = np.array([a, a, a])
    brr = np.array([b, b, b])
    # arr = np.array(a)
    # brr = np.array(b)
    r = EvalUtils.RMSE(arr, brr)
    print(r)

def test_pr_topk():
    a = [90, 80, 0, 100, 20, 0, 50]
    b = [70, 80, 100, 0, 30, 0, 70]

    arr = np.array(a)
    brr = np.array(b)
    aid = EvalUtils.value2id(arr, 60, False)
    bid = EvalUtils.value2id(brr, 60, False)
    print(aid)
    print(bid)
    ptopk = EvalUtils.precision(aid, bid, 3)
    print(ptopk)
    rtopk = EvalUtils.recall(aid, bid, 3)
    print(rtopk)

def test_rank_corr():
    a = [86, 97, 99, 100, 101, 103, 106, 110, 112, 113]
    b = [0, 20, 28, 27, 50, 29, 7, 17, 6, 12]

    arr = np.array([a, b, a])
    brr = np.array([b, a, b])
    ra = EvalUtils.get_rank(arr)
    rb = EvalUtils.get_rank(brr)
    print(ra)
    print(rb)
    rho = EvalUtils.rank_correlation(arr, brr)
    print(rho)  # [−0.175757575] * 3


def test_gather_scatter():
    a = [1,2,3,4,5]
    b = [3,4,5,6,7]
    m = [1,0,1,0,1]
    arr = np.array(a)
    brr = np.array(b)
    mrr = np.array(m)
    ag, bg = EvalUtils.gather(mrr, arr, brr)
    print(ag)
    print(bg)
    arr2, brr2 = EvalUtils.scatter(mrr, ag, bg)
    print(arr2)
    print(brr2)



def test_evalute():
    #Assuming 2 users, 7 items
    a = [90, 80, 0, 100, 20, 0, 50]
    b = [70, 80, 100, 0, 30, 0, 70]
    c = [100, 88, 0, 0, 0, 20, 0]
    d = [90, 70, 80, 0, 0, 0, 0]
    m1 = [1, 0, 1, 0, 1, 0, 1]
    m2 = [1, 1, 1, 1, 0, 0, 0]

    arr = np.array([a, c])
    brr = np.array([b, d])
    mrr = np.array([m1, m2])
    evals = evaluate(arr, brr, mrr)
    print(evals)


if __name__ == "__main__":
    print(f"This is {__file__}")
    # test_gather_scatter()
    test_evalute()