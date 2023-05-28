import numpy as np
from typing import Union, Tuple, List
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import math
class EvalUtils:

    @staticmethod
    def RMSE(y_hat: np.ndarray,
             y: np.ndarray,
             keepdims: bool = False) -> Union[np.ndarray, np.float64]:
        """root mean square error

        Args:
            y_hat (np.ndarray): Prediected values. B*N or N
            y (np.ndarray): True values. B*N or N
            keepdims (bool, optional): Whether to keep dims. Defaults to False.

        Returns:
            Union[np.ndarray, np.float64]: error(s), B or one float.
        """
        assert (y_hat.shape == y.shape
                ), f"{y_hat.shape} and {y.shape} is not same!"
        return np.sqrt(
            np.sum((y_hat - y) ** 2, axis=-1, keepdims=keepdims) / y_hat.shape[-1])

    @staticmethod
    def value2id(y: np.ndarray,
                 threshold: int = 80,
                 sort: bool = True) -> np.ndarray:
        """Get indexes by values.

        Args:
            y (np.ndarray): values. N
            threshold (int, optional): the values that less than `threshold` will be ignored. Defaults to 80.
            sort (bool, optional): sort before threshold descendingly. Defaults to True.

        Returns:
            np.ndarray: indexes. M (M <= N).
        """
        assert (len(y.shape) == 1)
        if sort:
            args = np.argsort(-y, -1)
            N = y.shape[-1]
            upper = N
            for i in range(N):
                if y[args[i]] < threshold:
                    upper = i
                    break
            args = args[:upper]
            return args
        else:
            tmp = np.where(y >= threshold)[-1]
            return tmp

    @staticmethod
    def get_rank(x: np.ndarray, reverse: bool = True) -> np.ndarray:
        """Get rank of values

        Args:
            x (np.ndarray): values. B*N or N
            reverse (bool, optional): Is descending? Defaults to True.

        Returns:
            np.ndarray: _description_
        """
        tmp = np.argsort(-x, axis=-1) if reverse else np.argsort(x, axis=-1)
        rank = np.zeros_like(tmp)
        if len(x.shape) == 1:
            for rk in range(rank.shape[-1]):
                rank[tmp[rk]] = rk
        elif len(x.shape) == 2:
            xind = np.arange(tmp.shape[0])
            for rk in range(rank.shape[-1]):
                rank[xind, tmp[..., rk]] = rk
        else:
            assert (0)
        return rank

    @staticmethod
    def true_positive(yid_hat: np.ndarray,
                      yid: np.ndarray) -> Union[np.ndarray, int]:
        """Get Number of true positive

        Args:
            yid_hat (np.ndarray): Predicted indexes. B*P or P
            yid (np.ndarray): True indexes. B*Q or Q

        Returns:
            Union[np.ndarray, int]: #TP (of B batches)
        """
        if len(yid_hat.shape) == len(yid.shape) == 1:
            return len(set(yid_hat) & set(yid))
        elif len(yid_hat.shape) == len(yid.shape) == 2:
            return np.array([
                len(set(yid_hat[i]) & set(yid[i])) for i in range(len(yid))
            ], dtype=np.int64)

    @staticmethod
    def precision(yid_hat: np.ndarray,
                  yid: np.ndarray,
                  topk: int = 10) -> Union[np.ndarray, float]:
        """Precision@K
        refer to https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
        
        Args:
            yid_hat (np.ndarray): Predicted indexes. B*P or P
            yid (np.ndarray): True indexes. B*Q or Q
            topk (int): K. Defaults to 10.

        Returns:
            Union[np.ndarray, float]: Precision@K (of B batch).
        """
        yid_hat2 = yid_hat[..., :topk]
        if yid_hat2.shape[-1] == 0:
            return 0.0
        yid2 = yid[..., :topk]
        tp = EvalUtils.true_positive(yid_hat2, yid2)
        return tp / yid_hat2.shape[-1]

    @staticmethod
    def recall(yid_hat: np.ndarray,
               yid: np.ndarray,
               topk: int = 10) -> Union[np.ndarray, float]:
        """Recall@K
        It may be not useful because we cannot get all of relative items in pratice.
        
        Args:
            yid_hat (np.ndarray): Predicted indexes. B*P or P.
            yid (np.ndarray): True indexes. B*Q or Q
            topk (int): K. Defaults to 10.

        Returns:
            Union[np.ndarray, float]: Recall@K (of B batch).
        """
        yid2 = yid[..., :topk]
        if yid2.shape[-1] == 0:
            return 0.0
        yid_hat2 = yid_hat[..., :topk]
        tp = EvalUtils.true_positive(yid_hat2, yid2)
        return tp / yid2.shape[-1]

    # ERROR and useless
    # @staticmethod
    # def accuracy(y_hat: np.ndarray,
    #              y: np.ndarray,
    #              topk: int = 10) -> Union[np.ndarray, float]:
    #     """Precision at Top k (sorted by value descendingly, count by id).

    #     Args:
    #         y_hat (np.ndarray): Prediected values. B*N or N
    #         y (np.ndarray): True valus. B*N or N
    #         topk (int, optional): K. Defaults to 10.

    #     Returns:
    #         Union[np.ndarray, float]: Precision@K.
    #     """
    #     assert (y_hat.shape == y.shape
    #             ), f"{y_hat.shape} and {y.shape} is not same!"
    #     ind_hat = np.argsort(-y_hat, axis=-1)[:topk]
    #     ind = np.argsort(-y, axis=-1)[:topk]
    #     tp = np.sum(ind == ind_hat, axis=-1)
    #     topkp = tp / min(topk, y.shape[-1])
    #     return topkp

    @staticmethod
    def rank_correlation(
            y_hat: np.ndarray,
            y: np.ndarray,
            method: str = "spearman") -> Union[np.ndarray, np.float64]:
        """Calculate Rank Correlation
        refer to https://en.wikipedia.org/wiki/Rank_correlation

        Args:
            y_hat (np.ndarray): the samples of first distribution. B*N or N
            y (np.ndarray): the samples of second distribution. B*N or N
            method (str, optional): ["spearman", "kendall"]. Defaults to "spearman".

        Returns:
            Union[np.ndarray, np.float64]: correlation(s). B or a float.
        """
        assert (y_hat.shape == y.shape
                ), f"{y_hat.shape} and {y.shape} is not same!"
        N = y_hat.shape[-1]
        if method == "spearman":
            # refer to https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
            rank1 = EvalUtils.get_rank(y_hat)
            rank2 = EvalUtils.get_rank(y)
            diff = np.sum((rank1 - rank2) ** 2, axis=-1)
            rho = 1 - (6 * diff) / (N * (N ** 2 - 1))
            return rho
        elif method == "kendall":
            # refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
            assert (0), "not implement yet"
        else:
            assert (0), f"no such method {method}"

    @staticmethod
    def gather(mask: np.ndarray, *args) -> List[np.ndarray]:
        """Gather `args` by the same `mask`
        """
        assert (len(mask.shape) == 1)
        rtns = []
        for y in args:
            assert (len(y.shape) == 1)
            assert (y.shape == mask.shape)
            rtns.append(y[np.where(mask > 0)])
        return rtns

    @staticmethod
    def scatter(mask: np.ndarray, *args) -> List[np.ndarray]:
        """Scatter `args` by the same `mask`
        """
        assert (len(mask.shape) == 1)
        rtns = []
        for y in args:
            assert (len(y.shape) == 1)
            rtn = np.zeros_like(mask)
            idx = np.where(mask > 0)
            assert (len(idx[0]) == len(y))
            rtn[idx] = y
            rtns.append(rtn)
        return rtns


def test_RMSE(path1, path2):
    scores_results = {}  # 存储打分值的列表
    scores_standard = {}  # 存储标答的列表
    # 真值
    with open(path1, 'r') as file:
        lines = file.readlines()  # 逐行读取文件内容
        for line in lines:
            line = line.strip()  # 去除行末尾的换行符
            if '|' in line:
                user, rates_num = [int(i) for i in line.split('|')]  # 提取用户ID和评分数目
                scores_standard[user] = {}  # 为该用户创建一个空字典
                continue  # 忽略包含"|"符号的行
            else: 
                item, rate = [int(i) for i in line.split()]  # 提取物品ID和评分值
                scores_standard[user][item] = rate  # 将评分值添加到字典中
                # scores_results.append(score)  # 将打分值添加到列表中
    # 预测值
    with open(path2, 'r') as file:
        lines = file.readlines()  # 逐行读取文件内容
        for line in lines:
            line = line.strip()  # 去除行末尾的换行符
            if '|' in line:
                user, rates_num = [int(i) for i in line.split('|')]  # 提取用户ID和评分数目
                scores_results[user] = {}  # 为该用户创建一个空字典
                continue  # 忽略包含"|"符号的行
            else:
                item, rate = line.split()  # 提取物品ID和评分值
                item = int(item)
                rate = float(rate)
                scores_results[user][item] = rate  # 将评分值添加到字典中
                # scores_standard.append(score)  # 将打分值添加到列表中
    result_true = []
    result_pred = []
    for user, _ in scores_standard.items():
        for item, score in scores_standard[user].items():
            result_true.append(int(score))
            result_pred.append(float(scores_results[user][item]))
    # a = [90, 80, 0, 100, 20, 0, 0]
    # b = [70, 80, 100, 0, 30, 0, 0]
    # arr = np.array([a, a, a])
    # brr = np.array([b, b, b])
    print(result_true[:10], "\n", result_pred[:10])
    print(len(result_true))
    print(len(result_pred))
    arr = np.array(scores_results)
    brr = np.array(scores_standard)
    # r = EvalUtils.RMSE(arr, brr)
    r = math.sqrt(mean_squared_error(result_true, result_pred))
    print(r)

if __name__ == "__main__":
    print(f"This is {__file__}")
    # test_gather_scatter()
    test_RMSE("./Data/train_test.txt", "./CF_item/Save/train_test_result.txt")
    # test_evalute()