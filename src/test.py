from datetime import datetime
import numpy as np
import graph_class
import pickle


class Tester:
    def __init__(self, base_path, build_new_graph=True, graphs=[]):
        self.data = self.read_fvecs(base_path)
        self.build_new_graph = build_new_graph
        self.graphs = graphs
        pass

    def test_graph_construction(self):
        print()
        print("*** Graph Construction ***")
        print("Testing set neighbor construction...")
        start = datetime.now()

        # Build with set neighbors
        sn_graph = graph_class.Graph(type="nsw-greedy", data=self.data)
        sn_graph.build_with_set_neighbors(index_factors=sn_graph.data)

        end = datetime.now()
        print("Time taken: ", end - start)
        print()

        print("Testing thresholding construction...")
        start = datetime.now()

        # Build with thresholding
        threshold_graph = graph_class.Graph(type="nsw-threshold", data=self.data)
        sn_graph.build_with_set_neighbors(index_factors=threshold_graph.data)

        end = datetime.now()
        print("Time taken: ", end - start)
        print()
        self.graphs.append(sn_graph)
        self.graphs.append(threshold_graph)

    def test_graph_search(self):
        pass

    def test_all(self):
        if self.build_new_graph:
            self.test_graph_construction()
        self.test_graph_search()

    def dump_graphs(self):
        for graph in self.graphs:
            ms = str(datetime.now().timetuple()[6])
            pickle.dump(graph, open("/graphs/" + ms, "wb"))

    def read_fvecs(self, fp):
        a = np.fromfile(fp, dtype="int32")
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def main():
    path = "data/siftsmall/siftsmall_base.fvecs"
    tester = Tester(path)
    tester.test_all()


if __name__ == "__main__":
    main()
