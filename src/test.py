from datetime import datetime
import numpy as np
import graph_class


class Tester:
    def __init__(self, base_path):
        self.data = self.read_fvecs(base_path)
        pass

    def test_graph_construction(self):
        print()
        print("*** Graph Construction ***")
        print("Testing set neighbor construction...")
        start = datetime.now()

        # Build with set neighbors
        sn_graph = graph_class.Graph(type="nsw-greedy", data=self.data)

        end = datetime.now()
        print("Time taken: ", end - start)
        print()

        print("Testing thresholding construction...")
        start = datetime.now()

        # Build with thresholding
        threshold_graph = graph_class.Graph(type="nsw-threshold", data=self.data)

        end = datetime.now()
        print("Time taken: ", end - start)
        print()

    def test_graph_search(self):
        pass

    def test_all(self):
        self.test_graph_construction()
        self.test_graph_search()

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
