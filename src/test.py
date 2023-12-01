from datetime import datetime
import numpy as np
import graph_class
import pickle
from tqdm import tqdm


class Tester:
    def __init__(self, base_path, graphs=[]):
        self.data = self.read_fvecs(base_path)
        self.build_new_graph = True if not graphs else False
        self.graphs = graphs
        pass

    def test_graph_construction(self):
        print()
        print("*** Graph Construction ***")
        print("Testing set neighbor construction...")
        start = datetime.now()

        # Build with set neighbors
        sn_graph = graph_class.Graph(
            type="nsw-greedy", data=self.data, k=5, m=1, build_with_thresholding=False
        )

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
        print("Time taken: ", end - start)
        print()
        self.graphs.append(sn_graph)
        # self.graphs.append(threshold_graph)

    def test_graph_search(self):
        print()
        print("*** Graph Search ***")
        print("Testing graph search...")
        print()
        ground_truth = self.read_ivecs("data/siftsmall/siftsmall_groundtruth.ivecs")
        query = self.read_fvecs("data/siftsmall/siftsmall_query.fvecs")
        k = 10

        results = []

        for graph in self.graphs:
            graph_dict = graph.graph
            results_greedy = []
            results_beam = []
            start_time = datetime.now()
            for q in tqdm(query):
                g = [r[1] for r in graph.greedy_search(graph_dict, q, k=k)[0]]
                b = [r[1] for r in graph.beam_search(graph_dict, q, k=k)[0]]
                results_greedy.append(g)
                results_beam.append(b)
            end_time = datetime.now()
            results.append((results_greedy, results_beam, end_time - start_time))

        true = ground_truth[:, :k]

        for i, result in enumerate(results):
            print("Graph #: ", i)
            greedy, beam, time_taken = result
            print("Greedy Length: ", len(greedy))
            print("Beam Length: ", len(beam))
            greedy_recall = self.calculate_recall(greedy, true)
            beam_recall = self.calculate_recall(beam, true)
            print("Greedy Recall: ", greedy_recall)
            print("Beam Recall: ", beam_recall)
            print("Time taken: ", time_taken)

        print()

    def test_all(self):
        if self.build_new_graph:
            self.test_graph_construction()
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
        for graph in self.graphs:
            ms = str(datetime.now().timetuple()[6])
            pickle.dump(graph, open("/graphs/" + ms, "wb"))

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
    with open("graphs/graph-set-k5-m1.pkl", "rb") as f:
        graph = pickle.load(f)
        graph_list = [graph]
        tester = Tester(path, graphs=graph_list)
        tester.test_all()


if __name__ == "__main__":
    main()
