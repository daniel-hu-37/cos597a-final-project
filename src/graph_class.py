import numpy as np
from scipy.spatial import distance

import heapq
import random
from typing import List, Tuple
from tqdm import tqdm


class Node:
    """
    Node for a navigable small world graph.

    Parameters
    ----------
    idx : int
        For uniquely identifying a node.

    value : 1d np.ndarray
        To access the embedding associated with this node.

    neighborhood : set
        For storing adjacent nodes.

    References
    ----------
    https://book.pythontips.com/en/latest/__slots__magic.html
    https://hynek.me/articles/hashes-and-equality/
    """

    __slots__ = ["idx", "value", "neighborhood"]

    def __init__(self, idx, value):
        self.idx = idx
        self.value = value
        self.neighborhood = set()

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.idx == other.idx


class Graph:
    def __init__(
        self,
        data,
        graph=None,
        build_k: int = 5,
        build_m: int = 10,
        greedy=True,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            An array of high-dimensional data points. Each data point is used to create a node in the NSW graph.

        graph : dict(int : Node)
            A dictionary of nodes, where each node is represented by a Node object. The key is the index of the node.

        Notes
        -----
        - The greedy search method used for finding nearest neighbors is dependent on the implementation
            of the `greedy_search` method in the same class.
        - The function provides a progress bar for tracking the graph construction process, using `tqdm`.
        - Bi-directional connections are established between each node and its neighbors.
        """
        if not graph:
            self.graph = self.build(data, build_k, build_m, greedy)
        else:
            self.graph = graph
        # self.size = len(list(self.graph.keys()))

    def build(
        self, index_factors: np.ndarray, k: int = 5, m: int = 10, greedy=True
    ) -> dict[int:Node]:
        """
        Builds a Navigable Small World (NSW) graph using a greedy approach.

        This function iteratively adds nodes to the graph. Each node is connected to its k nearest neighbors
        in the graph based on the distance between their respective high-dimensional data points.
        For the first k nodes, they are connected to all previously added nodes.
        The function uses a greedy search to find the nearest neighbors for each node.

        Parameters
        ----------
        index_factors : np.ndarray
            An array of high-dimensional data points. Each data point is used to create a node in the NSW graph.

        k : int
            The number of nearest neighbors to connect to each node in the graph.

        Returns
        -------
        List[Node]
            A list of Node objects, representing the nodes in the NSW graph. Each node contains information
            about its index, its high-dimensional value, and a set of indices representing its neighbors.

        Notes
        -----
        - The greedy search method used for finding nearest neighbors is dependent on the implementation
            of the `greedy_search` method in the same class.
        - The function provides a progress bar for tracking the graph construction process, using `tqdm`.
        - Bi-directional connections are established between each node and its neighbors.

        Examples
        --------
        >>> graph = Graph(type='nsw-greedy', data=some_data_array)
        >>> nsw_graph = graph.build_nsw_greedy(index_factors=some_data_array, k=5)
        """
        tqdm_loader = tqdm(index_factors)
        tqdm_loader.set_description("Building Graph")
        graph = {}
        for i, value in enumerate(tqdm_loader):
            node = Node(i, value)
            # if we already have more than k nodes in the graph, attach to the
            # k nearest neihgbors, found by greedy search
            if i > k:
                if greedy:
                    neighbors, _ = self.greedy_search(graph, node.value, k, m)
                else:
                    neighbors, _ = self.beam_search(graph, node.value, k, m)
                neighbors_indices = [node_idx for _, node_idx in neighbors]
            else:
                neighbors_indices = list(range(i))

            # insert bi-directional connection
            node.neighborhood.update(neighbors_indices)
            graph[i] = node
            for i in neighbors_indices:
                graph[i].neighborhood.add(node.idx)

        return graph

    ###########################################################################
    # def build_with_thresholding(
    #     self, index_factors: np.ndarray, threshold=0.5
    # ) -> dict[int:Node]:
    #     tqdm_loader = tqdm(index_factors)
    #     tqdm_loader.set_description("Building Graph")
    #     graph = {idx: Node(idx, val) for idx, val in enumerate(tqdm_loader)}

    #     # initialize each node to be connected to the next node so that the
    #     # graph is fully connected
    #     keys = list(graph.keys())
    #     for key in keys[:-1]:
    #         node = graph[key]
    #         node.neighborhood.add(key + 1)
    #     graph[keys[-1]].neighborhood.add(0)

    #     tqdm_loader_2 = tqdm(keys)
    #     tqdm_loader_2.set_description("This Vertex")
    #     for i, key in enumerate(tqdm_loader_2):
    #         for other_key in keys:
    #             node = graph[key]
    #             other = graph[other_key]
    #             if node.idx != other.idx:
    #                 # Calculate normalized difference
    #                 norm_diff = distance.cosine(node.value, other.value)

    #                 # Connect nodes if difference is below threshold
    #                 if norm_diff < threshold:
    #                     node.neighborhood.add(other.idx)
    #                     other.neighborhood.add(node.idx)

    #     return graph
    ###########################################################################

    def greedy_search(
        self, graph: List[Node], query: np.ndarray, k: int = 10, m: int = 1
    ) -> Tuple[List[Tuple[float, int]], float]:
        """
        Performs knn search using the navigable small world graph.

        Parameters
        ----------
        graph :
            Navigable small world graph from build_nsw_graph.

        query : 1d np.ndarray
            Query embedding that we wish to find the nearest neighbors.

        k : int
            Number of nearest neighbors returned.

        m : int
            The recall set will be chosen from m different entry points.

        Returns
        -------
        The list of nearest neighbors (distance, index) tuple.
        and the average number of hops that was made during the search.
        """
        result_queue = []
        visited_set = set()

        # hops = 0
        for _ in range(m):
            entry_node = random.randint(0, len(list(graph.keys())) - 1)
            result_queue, visited_set = self.single_it_greedy_search(
                graph, query, entry_node, k, result_queue, visited_set
            )

        # print("greedy visited: ", len(visited_set))
        # return heapq.nsmallest(k, result_queue), hops / m
        return heapq.nsmallest(k, result_queue), m

    def single_it_greedy_search(
        self, graph, query, entry_node, k, result_queue, visited_set
    ):
        entry_dist = distance.cosine(query, graph[entry_node].value)
        candidate_queue = []
        heapq.heappush(candidate_queue, (entry_dist, entry_node))

        temp_result_queue = []
        while candidate_queue:
            candidate_dist, candidate_idx = heapq.heappop(candidate_queue)
            # if candidate is further than the k-th (furthest) element from the result,
            # then we would break the repeat loop
            if len(temp_result_queue) >= k:
                current_k_dist, _ = heapq.nsmallest(k, temp_result_queue)[-1]
                if candidate_dist > current_k_dist:
                    break

            # iterate over neighbors and add them to candidates and results
            for friend_node in graph[candidate_idx].neighborhood:
                if friend_node not in visited_set:
                    visited_set.add(friend_node)

                    friend_dist = distance.cosine(query, graph[friend_node].value)
                    heapq.heappush(candidate_queue, (friend_dist, friend_node))
                    heapq.heappush(temp_result_queue, (friend_dist, friend_node))
                    # hops += 1

        result_queue = list(heapq.merge(result_queue, temp_result_queue))
        return result_queue, visited_set

    def beam_search(
        self,
        graph: List[Node],
        query: np.ndarray,
        k: int = 5,
        m: int = 1,
        beam_width: int = 2,
    ) -> Tuple[List[Tuple[float, int]], float]:
        """
        Performs knn search using beam search on the navigable small world graph.

        Parameters
        ----------
        graph :
            Navigable small world graph from build_nsw_graph.

        query : 1d np.ndarray
            Query embedding that we wish to find the nearest neighbors.

        k : int
            Number of nearest neighbors returned.

        m : int
            The recall set will be chosen from m different entry points.

        beam_width : int
            Number of nodes to consider at each level of the search.

        Returns
        -------
        The list of nearest neighbors (distance, index) tuple.
        and the average number of hops that was made during the search.
        """
        result_queue = []
        visited_set = set()

        # hops = 0
        for _ in range(m):
            entry_node = random.randint(0, len(list(graph.keys())) - 1)
            result_queue, visited_set = self.single_it_beam_search(
                graph, query, entry_node, k, result_queue, visited_set, beam_width
            )

        # print("beam visited: ", len(visited_set))
        # return heapq.nsmallest(k, result_queue), hops / m
        return heapq.nsmallest(k, result_queue), m

    def single_it_beam_search(
        self, graph, query, entry_node, k, result_queue, visited_set, beam_width
    ):
        entry_dist = distance.cosine(query, graph[entry_node].value)
        candidate_queue = []
        heapq.heappush(candidate_queue, (entry_dist, entry_node))

        temp_result_queue = []
        while candidate_queue:
            candidate_dist, candidate_idx = heapq.heappop(candidate_queue)

            if len(temp_result_queue) >= k:
                current_k_dist, _ = heapq.nsmallest(k, temp_result_queue)[-1]
                if candidate_dist > current_k_dist:
                    break

            beams, beams_rest = [candidate_idx], []
            for _ in range(min((beam_width - 1), len(candidate_queue) - 1)):
                _, cand_idx = heapq.heappop(candidate_queue)
                beams_rest.append(cand_idx)
            beams.extend(beams_rest)

            for idx in beams:
                for friend_node in graph[idx].neighborhood:
                    if friend_node not in visited_set:
                        visited_set.add(friend_node)
                        friend_dist = distance.cosine(query, graph[friend_node].value)
                        heapq.heappush(candidate_queue, (friend_dist, friend_node))
                        heapq.heappush(temp_result_queue, (friend_dist, friend_node))
                        # hops += 1

        result_queue = list(heapq.merge(result_queue, temp_result_queue))
        return result_queue, visited_set

    def dynamic_beam_search(
        self,
        graph: List[Node],
        query: np.ndarray,
        k: int = 5,
        m: int = 1,
        beam_width: int = 2,
        terminate: int = 4,
    ) -> Tuple[List[Tuple[float, int]], float]:
        """
        Performs knn search using beam search on the navigable small world graph.

        Parameters
        ----------
        graph :
            Navigable small world graph from build_nsw_graph.

        query : 1d np.ndarray
            Query embedding that we wish to find the nearest neighbors.

        k : int
            Number of nearest neighbors returned.

        m : int
            The recall set will be chosen from m different entry points.

        beam_width : int
            Number of nodes to consider at each level of the search.

        Returns
        -------
        The list of nearest neighbors (distance, index) tuple.
        and the average number of hops that was made during the search.
        """
        result_queue = []
        visited_set = set()

        # hops = 0
        for _ in range(m):
            entry_node = random.randint(0, len(list(graph.keys())) - 1)
            result_queue, visited_set = self.single_it_dynamic_search(
                graph,
                query,
                entry_node,
                k,
                result_queue,
                visited_set,
                beam_width,
                terminate,
            )

        # print("beam visited: ", len(visited_set))
        # return heapq.nsmallest(k, result_queue), hops / m
        return heapq.nsmallest(k, result_queue), m

    def single_it_dynamic_search(
        self,
        graph,
        query,
        entry_node,
        k,
        result_queue,
        visited_set,
        beam_width,
        terminate,
    ):
        entry_dist = distance.cosine(query, graph[entry_node].value)
        candidate_queue = []
        heapq.heappush(candidate_queue, (entry_dist, entry_node))

        temp_result_queue = []
        counter = 0
        while candidate_queue:
            if counter > terminate:
                beam_width = 2
            candidate_dist, candidate_idx = heapq.heappop(candidate_queue)

            if len(temp_result_queue) >= k:
                current_k_dist, _ = heapq.nsmallest(k, temp_result_queue)[-1]
                if candidate_dist > current_k_dist:
                    break

            beams, beams_rest = [candidate_idx], []
            for _ in range(min((beam_width - 1), len(candidate_queue) - 1)):
                _, cand_idx = heapq.heappop(candidate_queue)
                beams_rest.append(cand_idx)
            beams.extend(beams_rest)

            for idx in beams:
                for friend_node in graph[idx].neighborhood:
                    if friend_node not in visited_set:
                        visited_set.add(friend_node)
                        friend_dist = distance.cosine(query, graph[friend_node].value)
                        heapq.heappush(candidate_queue, (friend_dist, friend_node))
                        heapq.heappush(temp_result_queue, (friend_dist, friend_node))
                        # hops += 1
            counter += 1

        result_queue = list(heapq.merge(result_queue, temp_result_queue))

        return result_queue, visited_set

    def random_init(self):
        return random.randint(0, len(list(self.graph.keys())) - 1)


###########################################################################
# def delete_node(self, idx):
#     """
#     Deletes a node from the graph.
#     """
#     # first remove all references to the node
#     node = self.graph[idx]
#     for neighbor in node.neighborhood:
#         self.graph[neighbor].neighborhood.remove(idx)

#     # then delete from graph
#     del self.graph[idx]
###########################################################################
