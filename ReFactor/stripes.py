from operator import imod
import pickle
import numpy as np
import heapq
import math
import json
import getopt
import sys
import os

from config import *
from utils import save_result, time_test, read_data, int2bytes, bytes2int, setup

# DEBUG = True

class SubGraph:
    """
    Use dict rather than list to save memory
    """
    def __init__(self) -> None:
        self.in_edges = {}
        self.out_degrees = {}
        self.src_nodes = set()

    def add_edges(self, edges):
        for edge in edges:
            srcs = self.in_edges.get(edge[1], [])
            srcs.append(edge[0])
            self.in_edges[edge[1]] = srcs
            self.out_degrees[edge[0]] = self.out_degrees.get(edge[0], 0) + 1
        # sort by key
        self.in_edges = dict(sorted(self.in_edges.items(), key=lambda x: x[0]))
        self.out_degrees = dict(sorted(self.out_degrees.items(), key=lambda x: x[0]))
        # for each key sort by value
        for dst in self.in_edges:
            self.in_edges[dst].sort()

    def save(self, sgid:int):
        """
        The structure of the sgraph data file:
        <len(dsts)> <dsts[0]> <dsts[1]> ...
        <2 * len(srcs)> <src0> <outd0> <src1> <outd1> ...
        <len(srcs0)> <srcs0[0]> <srcs0[1]> ...
        <len(srcs1)> <srcs0[0]> <srcs0[1]> ...
        ...
        """
        nbytes = 0
        filename = os.path.join(SGRAPH_PATH, f"sgraph_{sgid}.data")
        with open(filename, 'wb') as f:
            f.write(int2bytes(len(self.in_edges.keys())))
            for dst in self.in_edges.keys():
                nbytes += f.write(int2bytes(dst))
            f.write(int2bytes(2 * len(self.out_degrees.keys())))
            for src, outd in self.out_degrees.items():
                nbytes += f.write(int2bytes(src))
                nbytes += f.write(int2bytes(outd))
            for srcs in self.in_edges.values():
                nbytes += f.write(int2bytes(len(srcs)))
                for src in srcs:
                    nbytes += f.write(int2bytes(src))

        print(f"Finish to write {nbytes} bytes to {filename}")

    def load(self, sgid:int):
        filename = os.path.join(SGRAPH_PATH, f"sgraph_{sgid}.data")
        with open(filename, 'rb') as f:
            ndsts = bytes2int(f.read(SIZE_INT))
            dsts = []
            for _ in range(ndsts):
                dsts.append(bytes2int(f.read(SIZE_INT)))
            # f.write(int2bytes(2 * len(self.out_degrees.keys())))
            noutds = bytes2int(f.read(SIZE_INT)) // 2
            outds = {}
            for _ in range(noutds):
                node = bytes2int(f.read(SIZE_INT))
                outd = bytes2int(f.read(SIZE_INT))
                outds[node] = outd
            inedges = {}
            for dst in dsts:
                nsrcs = bytes2int(f.read(SIZE_INT))
                srcs = []
                for _ in range(nsrcs):
                    srcs.append(bytes2int(f.read(SIZE_INT)))
                inedges[dst] = srcs
        self.__init__()
        # should be in sequence
        self.in_edges = inedges
        self.out_degrees = outds
        
    
    def load_from_block(self, block_id:int):
        block_file = os.path.join(BLOCK_PATH, f"block_{block_id}.data")
        self.in_edges = {}
        self.out_degrees = {}
        self.src_nodes = set()
        with open(block_file, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            length = fp.tell()
            fp.seek(0, os.SEEK_SET)
            while(fp.tell() < length):
                offset = bytes2int(fp.read(SIZE_INT))
                dst = bytes2int(fp.read(SIZE_INT))
                srcs = [ bytes2int(fp.read(SIZE_INT)) for _ in range(offset - 1)]
                self.src_nodes |= set(srcs)  # TODO: optimize
                # assert(self.in_edges.get(dst) is None)
                self.in_edges[dst] = srcs 
        self.src_nodes = list(self.src_nodes)
        self.src_nodes.sort()
        ndsts = len(self.in_edges.keys())
        # print(f"Finish load from {block_file}, number of dsts: {ndsts}")       
        return ndsts


def load_outds_from_sgraph(sgraph_id:int, start, end) -> np.ndarray:
    """
    load out degrees of points in [`start`, `end`)
    :return: a list in length (end - start), fill 0 if out degree is 0
    """
    filename = os.path.join(SGRAPH_PATH, f"sgraph_{sgraph_id}.data")
    ans = np.zeros(end - start, dtype=int)
    with open(filename, "rb") as fp:
        offset = bytes2int(fp.read(SIZE_INT))  # len(dsts)
        fp.seek(offset * SIZE_INT, os.SEEK_CUR)
        offset = bytes2int(fp.read(SIZE_INT))  # 2 * len(srcs)
        for _ in range(offset // 2):
            src = bytes2int(fp.read(SIZE_INT))
            outd = bytes2int(fp.read(SIZE_INT))
            if src >= end:
                break
            if src >= start:
                ans[src - start] = outd
    return ans

def load_outds_by_range(start:int, end:int) -> list:
    """
    """
    filename = os.path.join(BLOCK_PATH, f"outd.data")
    length = end - start
    ans = [0] * length
    with open(filename, "rb") as fp:
        fp.seek((start - 1) * SIZE_INT, os.SEEK_SET)  # outd.data begins from 1!
        for i in range(length):
            ans[i] = bytes2int(fp.read(SIZE_INT))
    return ans
        

def load_srcs_by_dst(sgraph_id:int, dst:int) -> list:
    """
    load only one line of srcs that point to `dst`
    """
    srcs = []
    filename = os.path.join(SGRAPH_PATH, f"sgraph_{sgraph_id}.data")
    with open(filename, "rb") as fp:
        offset = bytes2int(fp.read(SIZE_INT))  # len(dsts)
        for i in range(offset):
            cur = bytes2int(fp.read(SIZE_INT))
            if cur < dst:
                continue
            elif cur > dst:
                break
            else:
                fp.seek((offset - i - 1) * SIZE_INT, os.SEEK_CUR)
                offset = bytes2int(fp.read(SIZE_INT)) 
                fp.seek(offset * SIZE_INT, os.SEEK_CUR)
                for _ in range(i):
                    offset = bytes2int(fp.read(SIZE_INT)) 
                    fp.seek(offset * SIZE_INT, os.SEEK_CUR)
                offset = bytes2int(fp.read(SIZE_INT))
                for _ in range(offset):
                    srcs.append(bytes2int(fp.read(SIZE_INT)))
                break
    return srcs

def save_ginfo(ginfo:dict):
    ginfo_fname = os.path.join(BLOCK_PATH, "ginfo.json")
    with open(ginfo_fname, "w") as fp:
        json.dump(ginfo, fp)

def load_ginfo() -> dict:
    ginfo_fname = os.path.join(BLOCK_PATH, "ginfo.json")
    with open(ginfo_fname, "r") as fp:
        global_info = json.load(fp)
    return global_info

def prepare_datas() -> dict:
    """
    1. Read from datasets and write it to disk in some format batch by batch (get sgraph_xx.data)
    2. Read special lines from each sgraph_xx.data and do multi-line merge, then write to disk again (get block_xx.data)
    3. Read special range of nodes from each sgraph_xx.data and do multi-line add, then write to disk again (get outd.data)
    4. Get global infos including total number of nodes

    The strucure of block_xx.data:\\
    <len(srcs0) + 1> <dst0> <srcs0[0]> <srcs0[1]> ...
    ...

    The structure of outd.data is a sequence of out degrees of every node, begins from node 1!
    """
    global_infos = {}
    N = 0  # total number of nodes
    M = 0  # total number of sgraph
    # Step 1
    for edges, nnodes in read_data(DATA_IN, CHUNK_SIZE):
        sgraph = SubGraph()
        N = max(nnodes, N)
        sgraph.add_edges(edges)
        sgraph.save(M)
        M += 1

    # Step 2
    nwrite = 0
    nblock = 0  # total
    block_file = os.path.join(BLOCK_PATH, "block_0.data")
    block_fp = open(block_file, "wb")
    for dst in range(1, N + 1):
        ordered_src_lists = []
        # len_srcs = 0

        for i in range(M):
            srcs = load_srcs_by_dst(i, dst)
            if len(srcs) > 0:
                ordered_src_lists.append(srcs)
                # len_srcs += len(srcs)
        mlist = list(heapq.merge(*ordered_src_lists)) # considering repeat edge? 
        len_srcs = len(mlist)
        if len_srcs == 0:
            continue
        block_fp.write(int2bytes(len_srcs + 1))
        block_fp.write(int2bytes(dst))
        for r in mlist:
            nwrite += 1
            block_fp.write(int2bytes(r))

        # ignore the extreme case: Just length of **one** merged list is far more than CHUNK_SIZE
        if nwrite >= CHUNK_SIZE:
            block_fp.close()
            print(f"write {nwrite} srcs to {block_file}")
            nblock += 1
            nwrite = 0
            block_file = os.path.join(BLOCK_PATH, f"block_{nblock}.data")
            block_fp = open(block_file, "wb")
    
    if nwrite > 0:
        block_fp.close()
        print(f"write {nwrite} srcs to {block_file}")
        nblock += 1

    # Step 3
    outdfile = os.path.join(BLOCK_PATH, "outd.data")
    with open(outdfile, "wb") as outd_fp:
        for i in range(math.ceil(N / CHUNK_SIZE)):
            start = i * CHUNK_SIZE + 1
            end = min((i + 1) * CHUNK_SIZE, N) + 1
            sumd = np.zeros((end - start, ), dtype=int)
            for j in range(M):
                sumd += load_outds_from_sgraph(j, start, end)
            for s in sumd.tolist():
                outd_fp.write(int2bytes(s))

    global_infos["N"] = N
    global_infos["M"] = M
    global_infos["nblock"] = nblock
    save_ginfo(global_infos)
    
    print(f"Prepare Finished! total number of nodes: {N}, total number of sgraphs: {M}, total number of blocks: {nblock}")
    return global_infos


def save_rank(rank:np.ndarray, rid:int, cate:bool) -> None:
    catename = "old" if cate else "new"
    filename = os.path.join(RANK_PATH, f"rank_{catename}_{rid}.data")
    with open(filename, "wb") as fp:
        # print(f"save rank file {filename}")
        pickle.dump(rank.tolist(), fp)

def load_rank(rid:int, cate:bool, default=None, length=None) -> np.ndarray:
    catename = "old" if cate else "new"
    filename = os.path.join(RANK_PATH, f"rank_{catename}_{rid}.data")
    tmp = []
    if not os.path.exists(filename):
        if default is None or length is None:
            assert(0), f"{filename} is not exist!"
        # print(f"load rank_{catename}_{rid} full of {default}")
        return np.full(length, default, dtype=np.float32)
    with open(filename, "rb") as fp:
        # print(f"load rank from file {filename}")
        tmp = pickle.load(fp)
    return np.array(tmp)

def remove_rank(rid:int, cate:bool):
    catename = "old" if cate else "new"
    filename = os.path.join(RANK_PATH, f"rank_{catename}_{rid}.data")
    os.remove(filename)
    # print(f"{filename} is removed.")

def update_rank(rank:np.ndarray, rid:int, cate:bool):
    """
    add `rank` to No.`rid` rank data files 
    """
    r = load_rank(rid, cate, 0, len(rank))
    r += rank
    save_rank(r, rid, cate)
    return r


def update_rank_d(rank_d:dict, cate:bool, nrank, N) -> None:
    """
    update to get r_new, `cate` should be current r_new's cate.
    """
    klist = list(rank_d.keys())
    mink = klist[0]
    maxk = klist[-1]
    # assert(mink == min(klist))
    # assert(maxk == max(klist))
    rid1 = mink // CHUNK_SIZE
    rid2 = maxk // CHUNK_SIZE
    s1 = rid1 * CHUNK_SIZE + 1
    s2 = rid2 * CHUNK_SIZE + 1
    if rid1 == rid2:
        if rid1 == nrank - 1:
            r = np.zeros(N - rid1 * CHUNK_SIZE)
        else:
            r = np.zeros(CHUNK_SIZE)
        for k, v in rank_d.items():
            r[k - s1] = v
        update_rank(r, rid1, cate)
    elif rid1 + 1 == rid2:
        r1 = np.zeros(CHUNK_SIZE)
        if rid2 == nrank - 1:
            r2 = np.zeros(N - rid2 * CHUNK_SIZE)
        else:
            r2 = np.zeros(CHUNK_SIZE)
        i = 0
        for k, v in rank_d.items():
            if k - s1 < CHUNK_SIZE:
                r1[k - s1] = v
            else:
                r2[k - s2] = v
        update_rank(r1, rid1, cate)
        update_rank(r2, rid2, cate)
    else:
        # impossible case because a block has no more than CHUNK_SIZE dsts.
        assert(0), f"rid1: {rid1}, rid2{rid2}, rank_d: {rank_d}"

    
def pagerank(global_info:dict):
    N = global_info["N"]
    nblock = global_info["nblock"]
    nrank = math.ceil(N / CHUNK_SIZE)
    iter = 0
    outdfile = os.path.join(BLOCK_PATH, "outd.data")
    last_is_old = True
    # init
    for j in range(nrank):
        if j == nrank - 1:
            r_old = load_rank(j, last_is_old, 1 / N, N - j * CHUNK_SIZE)
        else:
            r_old = load_rank(j, last_is_old, 1 / N, CHUNK_SIZE)
        save_rank(r_old, j, last_is_old)
        
    # do until converage
    while True:
        sum_rnew = 0.0  # global sum of r_new
        sum_e = 0.0  # global sum of absulute error
        total_ndsts = 0  # total number of ndsts
        
        # calculate in M blocks 
        for i in range(nblock):
            sgraph = SubGraph()
            ndsts = sgraph.load_from_block(i)
            dsts = sgraph.in_edges.keys()
            total_ndsts += ndsts
            r_new_d = {k : 0 for k in dsts}
            tags = {k : 0 for k in dsts}  #  there are still srcs beginning from tag[dst] which point to dst have not been used to calculate r_new_d[dst]
            for j in range(nrank):
                start = j * CHUNK_SIZE + 1
                end = min(((j + 1) * CHUNK_SIZE), N) + 1
                outds = load_outds_by_range(start, end)
                # r_old = load_rank(j, 1/N, end-start)
                r_old = load_rank(j, last_is_old)
                for dst in dsts:
                    # dind = dmap[dst]
                    srcs = sgraph.in_edges[dst]  # ascending list
                    for k in range(tags[dst], len(srcs)):
                        tags[dst] += 1
                        src = srcs[k]
                        if src >= end:
                            break
                        r_new_d[dst] += TELEPORT * r_old[src - start] / outds[src - start]
                    
            # for val in r_new_d.values():
            #     sum_rnew += val
            sum_rnew += sum(r_new_d.values())
            # 
            # for key in r_new_d.keys():
            #     r_new_d[key] += S  # TODO: not sure
            # sum_e += np.linalg.norm(r_new - r_old, ord=NORM)
            # save_rank(r_new, i)
            update_rank_d(r_new_d, not last_is_old, nrank, N)

        adder = np.full(CHUNK_SIZE, (1 - sum_rnew) / N)
        for j in range(nrank):
            if j == nrank - 1:
                adder = adder[:N - j * CHUNK_SIZE]
            r_new = update_rank(adder, j, not last_is_old)
            r_old = load_rank(j, last_is_old)
            sum_e += np.linalg.norm(r_new - r_old, ord=NORM)

        iter += 1
        print(f"absolute error: {sum_e}, iter: {iter}")
        if sum_e < EPSILON or iter >= MAX_ITER:
        # if iter >= MAX_ITER:
            print(f"absolute error: {sum_e}, iter: {iter}")
            break
        # delete useless rank file!
        for j in range(nrank):
            remove_rank(j, last_is_old)
        last_is_old = not last_is_old
    global_info["r_new_cate"] = last_is_old
    global_info["nrank"] = nrank

    
def save_dist_result(global_info:dict):
    """
    This is the only part which does not follow memory limits because it is just for show results.
    """
    N = global_info["N"]
    nrank = global_info["nrank"]
    r_new_cate = global_info["r_new_cate"]
    rlist = []
    with open(STRIPE_OUT, "w", encoding="utf-8"):
        for rid in range(nrank):
            rlist.extend(load_rank(rid, r_new_cate).tolist())
    result = {}
    for i, r in enumerate(rlist):
        result[i + 1] = r
    save_result(result, STRIPE_OUT)
        


if __name__ == '__main__':
    print("This is %s" % 'Stripe Page Rank')

    pairs, _ = getopt.getopt(sys.argv[1:], "-r", ["remains"])
    
    remains_block = True if len(pairs) > 0 else False
    
    setup(True, remains_block)

    if not remains_block:
        global_info = time_test("pre_datas", prepare_datas)
    else:
        global_info = load_ginfo()

    time_test("pagerank", pagerank, global_info)

    save_dist_result(global_info)

    os.system("pause")
