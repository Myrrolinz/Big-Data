from typing import Union, Tuple, List
import numpy as np


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


def evaluate(y_hat,
             y,
             mask,
             save_ori=True,
             rmse=True,
             ptopk=True,
             rtopk=True,
             corr=True) -> dict:
    """A kind of usage of EvalUtils
    """
    evals = {}
    B = y.shape[0]
    ptopklist = []
    rtopklist = []
    rmselist = []
    corrlist = []
    for i in range(B):
        yh2, y2 = EvalUtils.gather(mask[i], y_hat[i], y[i])  # gathered
        yid_hat = EvalUtils.value2id(yh2, 80, True)
        yid = EvalUtils.value2id(y2, 80, True)
        if ptopk:
            ptopklist.append(EvalUtils.precision(yid_hat, yid, 10))
        if rtopk:
            rtopklist.append(EvalUtils.recall(yid_hat, yid, 10))
        if rmse:
            rmselist.append(EvalUtils.RMSE(yh2, y2))
        if corr:
            corrlist.append(EvalUtils.rank_correlation(yh2, y2, "spearman"))
    if ptopk:
        evals["ptopk"] = sum(ptopklist) / B
        if save_ori:
            evals["ptopk_ori"] = ptopklist
    if rtopk:
        evals["rtopk"] = sum(rtopklist) / B
        if save_ori:
            evals["rtopk_ori"] = rtopklist
    if rmse:
        evals["rmse"] = sum(rmselist) / B
        if save_ori:
            evals["rmse_ori"] = rmselist
    if corr:
        evals["corr"] = sum(corrlist) / B
        if save_ori:
            evals["corr_ori"] = corrlist
    return evals


def evaluate2(y_hat:List[List], y:List[List],
              rmse=True,
              ptopk=True,
              rtopk=True,
              corr=True,
              sort=True,
              thresh=80,
              k=10,
              save_ori=False) -> dict:
    assert(len(y_hat) == len(y))
    evals = {}
    B = len(y)
    ptopklist = []
    rtopklist = []
    rmselist = []
    corrlist = []
    for u in range(B):
        yh2 = np.array(y_hat[u])
        y2 = np.array(y[u])
        yid_hat = EvalUtils.value2id(yh2, thresh, sort)
        yid = EvalUtils.value2id(y2, thresh, sort)
        if ptopk:
            ptopklist.append(EvalUtils.precision(yid_hat, yid, k))
        if rtopk:
            rtopklist.append(EvalUtils.recall(yid_hat, yid, k))
        if rmse:
            rmselist.append(EvalUtils.RMSE(yh2, y2))
        if corr:
            corrlist.append(EvalUtils.rank_correlation(yh2, y2, "spearman"))
    if ptopk:
        evals["ptopk"] = sum(ptopklist) / B
        if save_ori:
            evals["ptopk_ori"] = ptopklist
    if rtopk:
        evals["rtopk"] = sum(rtopklist) / B
        if save_ori:
            evals["rtopk_ori"] = rtopklist
    if rmse:
        evals["rmse"] = sum(rmselist) / B
        if save_ori:
            evals["rmse_ori"] = rmselist
    if corr:
        evals["corr"] = sum(corrlist) / B
        if save_ori:
            evals["corr_ori"] = corrlist
    return evals


if __name__ == "__main__":
    print(f"This is {__file__}")
