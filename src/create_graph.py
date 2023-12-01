import numpy as np
from scipy.spatial import distance
import pickle
import heapq
import random
from typing import List, Tuple

from graph_class import Graph


# now define a function to read the fvecs file format of Sift1M dataset
def read_fvecs(fp):
    a = np.fromfile(fp, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def main():
    base = read_fvecs("../data/siftsmall/siftsmall_base.fvecs")  # 1M samples
    # also get some query vectors to search with
    # query = read_fvecs("./siftsmall/siftsmall_query.fvecs")
    # take just one query (there are many in sift_learn.fvecs)
    # xq = xq[0].reshape(1, xq.shape[1])

    g = Graph(type="nsw", data=base)

    k = 3
    print(g)
    graph = g.build_nsw_greedy(base, k)

    with open("graphk3.pkl", "wb") as f:
        pickle.dump(graph, f)


if __name__ == "__main__":
    main()
