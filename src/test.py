from datetime import datetime
import numpy as np
import graph_class
import pickle
from tqdm import tqdm
import time
import heapq


class Tester:
    def __init__(self, path, graphs=[]):
        self.data = self.read_fvecs(path)
        if graphs:
            self.graphs = graphs
        else:
            self.graphs = []

    def test_graph_construction(self):
        print()
        print("*** Graph Construction ***")
        print("Testing construction...")
        start = datetime.now()

        graph = graph_class.Graph(data=self.data)

        end = datetime.now()
        print("Time taken: ", end - start)
        print()

        # print("Testing thresholding construction...")
        # start = datetime.now()

        # # Build with thresholding
        # threshold_graph = graph_class.Graph(
        #     type="nsw-threshold",
        #     data=self.data,
        #     build_with_thresholding=True,
        #     threshold=0.3,
        # )

        # end = datetime.now()
        # print("Time taken: ", end - start)
        # print()
        self.graphs.append(graph)
        # self.graphs.append(threshold_graph)

    def test_graph_search(self):
        print()
        print("*** Graph Search ***")
        print("Testing graph search...")
        print()
        ground_truth = self.read_ivecs("data/siftsmall/siftsmall_groundtruth.ivecs")
        query = self.read_fvecs("data/siftsmall/siftsmall_query.fvecs")
        k = 5

        results = []

        for graph in self.graphs:
            graph_dict = graph.graph
            results_greedy = []
            results_beam = []
            greedy_time, beam_time = 0, 0
            for q in tqdm(query):
                random_init = graph.random_init()
                start_time = datetime.now()
                # normal greedy search
                # g = [r[1] for r in graph.greedy_search(graph_dict, q, k, m=1)[0]]

                # single iteration greedy search
                greedy_result = graph.single_it_greedy_search(
                    graph_dict, q, random_init, k, [], set()
                )[0]
                g = [r[1] for r in heapq.nsmallest(k, greedy_result)]

                end_time = datetime.now()
                greedy_time += (end_time - start_time).total_seconds()
                start_time = datetime.now()
                # normal beam search
                # b = [r[1] for r in graph.beam_search(graph_dict, q, k, m=1)[0]]

                # single iteration beam search
                beam_result = graph.single_it_beam_search(
                    graph_dict, q, random_init, k, [], set(), beam_width=5
                )[0]
                b = [r[1] for r in heapq.nsmallest(k, beam_result)]

                end_time = datetime.now()
                beam_time += (end_time - start_time).total_seconds()
                results_greedy.append(g)
                results_beam.append(b)
            results.append((results_greedy, results_beam))

        true = ground_truth[:, :k]

        for i, result in enumerate(results):
            print("Graph #: ", i)
            greedy, beam = result
            greedy_recall = self.calculate_recall(greedy, true)
            beam_recall = self.calculate_recall(beam, true)
            print("Greedy Recall: ", self.truncate(greedy_recall, 3))
            print("Beam Recall: ", self.truncate(beam_recall, 3))
            print("Greedy Time: ", self.truncate(greedy_time, 3))
            print("Beam Time: ", self.truncate(beam_time, 3))
        print()

    def truncate(self, num, dec):
        return str(num)[: str(num).find(".") + dec + 1]

    def test_all(self):
        if not self.graphs:
            self.test_graph_construction()
            self.dump_graphs()
        self.test_graph_search()

    def calculate_recall(self, predicted_neighbors, actual_neighbors):
        total_recall = 0

        for pred, actual in zip(predicted_neighbors, actual_neighbors):
            true_positives = len(set(pred) & set(actual))
            possible_positives = len(set(actual))

            recall = true_positives / possible_positives if possible_positives else 0

            total_recall += recall

        average_recall = total_recall / len(actual_neighbors)

        return average_recall

    def dump_graphs(self):
        print()
        print("Dumping graphs immediately after construction...")
        print()
        for graph in self.graphs:
            ms = str(int(time.time() * 1000))
            with open("graphs/" + ms + ".pkl", "wb") as f:
                pickle.dump(graph, f)

    def read_fvecs(self, fp):
        a = np.fromfile(fp, dtype="int32")
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")

    def read_ivecs(self, fname):
        a = np.fromfile(fname, dtype="int32")
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()


def main():
    path = "data/siftsmall/siftsmall_base.fvecs"
    tester = Tester(path, graphs=[])
    tester.test_all()

    # full_construct = "graphs/1701966839512.pkl"
    # partial_construct = "graphs/3.pkl"
    # with open(full_construct, "rb") as f:
    #     graph = pickle.load(f)
    #     tester = Tester(path, graphs=[graph])
    #     tester.test_all()


if __name__ == "__main__":
    main()
