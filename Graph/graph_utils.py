import re
import ray
import copy
import math
import scipy
import psutil
import pickle
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing

from tqdm import tqdm
from ray.util import ActorPool
from scipy.spatial import Delaunay



def create_vertices(corpus: list, k: int) -> dict:
    '''
    Create vertices from vectorized corpus splitted by sentences

    Parameters
    ----------
    corpus : list
    Vectorized corpus = [document_1, ..., document_i ..., document_n], where
    text_i = [{'document_index': i,
               'sentence_index': 0,
               'sentence_text': {word: vector,
                                 word: vector, ...}},
              ...,
              {'document_index': i,
               'sentence_index': m_i,
               'sentence_text': {word: vector,
                                 word: vector, ...}}]
    So, we get n documents and m_i sentences for each document

    k : int
    Vector length

    Returns
    -------
    vertices : dict
    Returns dict = {word: Node(vector,
                               empty set)}
    '''
    file_ = open(f'create_vertices_progress.txt', 'w')
    vertices = dict()
    for text in tqdm(corpus, file=file_):
        for sentence in text:
            for word, vector in sentence['sentence_text'].items():
                if word not in vertices.keys():
                    try:
                        vertices[word] = Node(word=word,
                                              vector=np.array(vector[:k]),
                                              neighbors=set())
                    except:
                        pass
    file_.close()
    return vertices


def create_words_text_neighbors(corpus: list) -> dict:
    '''
    Create neighbors from vectorized corpus splitted by sentences

    Parameters
    ----------
    corpus : list
    Vectorized corpus = [document_1, ..., document_i ..., document_n], where
    text_i = [{'document_index': i,
               'sentence_index': 0,
               'sentence_text': {word: vector,
                                 word: vector, ...}},
              ...,
              {'document_index': i,
               'sentence_index': m_i,
               'sentence_text': {word: vector,
                                 word: vector, ...}}]
    So, we get n documents and m_i sentences for each document

    Returns
    -------
    neighbors : dict
    Returns dict = {word: WordNeighbors(word,
                                        document_neighbors,
                                        sentence_neighbors)}
    '''
    file_ = open(f'create_words_text_neighbors_progress.txt', 'w')
    words_text_neighbors = dict()
    for text in tqdm(corpus, file=file_):
        document_words = set()
        for sentence in text:
            document_words.update(set(sentence['sentence_text'].keys()))
        for sentence in text:
            for word, vector in sentence['sentence_text'].items():
                word_document_neighbors = document_words.difference(set([word]))
                word_sentence_neighbors = set(sentence['sentence_text'].keys()).difference(set([word]))
                if word in vertices.keys():
                    words_text_neighbors[word].document_neighbors.update(word_document_neighbors)
                    words_text_neighbors[word].sentence_neighbors.update(word_sentence_neighbors)
                else:
                    try:
                        words_text_neighbors[word] = WordNeighbors(word=word,
                                                                   document_neighbors=word_document_neighbors,
                                                                   sentence_neighbors=word_sentence_neighbors)
                    except:
                        pass
    file_.close()
    return words_text_neighbors

def convert_corpus_to_list_of_texts(corpus: list) -> list:
    '''
    Convert corpus to list of texts from vectorized corpus splitted by sentences

    Parameters
    ----------
    corpus : list
    Vectorized corpus = [document_1, ..., document_i ..., document_n], where
    text_i = [{'document_index': i,
               'sentence_index': 0,
               'sentence_text': {word: vector,
                                 word: vector, ...}},
              ...,
              {'document_index': i,
               'sentence_index': m_i,
               'sentence_text': {word: vector,
                                 word: vector, ...}}]
    So, we get n documents and m_i sentences for each document

    Returns
    -------
    texts : list
    Returns list of texts}
    '''
    texts = []
    file_ = open(f'convert_corpus_to_list_of_texts.txt', 'w')
    for text in tqdm(corpus, file=file_):
        document = []
        for sentence in text:
            for word, vector in sentence['sentence_text'].items():
                if re.findall(r"[А-Яа-я]+", word):
                    document.append(word)
        texts.append(' '.join(document))
    return texts


class WordNeighbors:
    def __init__(self, word: str, document_neighbors: set, sentence_neighbors: set):
        '''
        WordNeighbors initialization

        Parameters
        ----------
        word : str
        document_neighbors: set
        sentence_neighbors: set
        '''
        self.__word = copy.deepcopy(word)
        self.__document_neighbors = copy.deepcopy(document_neighbors)
        self.__sentence_neighbors = copy.deepcopy(sentence_neighbors)

    @property
    def word(self):
        return self.__word

    @word.setter
    def word(self, word: str):
        self.__word = copy.deepcopy(word)

    @property
    def document_neighbors(self):
        return self.__document_neighbors

    @document_neighbors.setter
    def document_neighbors(self, document_neighbors: set):
        self.__document_neighbors = document_neighbors

    @property
    def sentence_neighbors(self):
        return self.__sentence_neighbors

    @sentence_neighbors.setter
    def sentence_neighbors(self, sentence_neighbors: set):
        self.__sentence_neighbors = sentence_neighbors


'''
Node and Graph
'''

class Node:
    def __init__(self, word: str, vector: np.ndarray, neighbors: set):
        '''
        Node initialization

        Parameters
        ----------
        word : str
        vector: np.ndarray
        neighbors: set
        '''
        self.__word = copy.deepcopy(word)
        self.__vector = copy.deepcopy(vector)
        self.__neighbors = copy.deepcopy(neighbors)

    @property
    def word(self):
        return self.__word

    @word.setter
    def word(self, word: str):
        self.__word = copy.deepcopy(word)

    @property
    def vector(self):
        return self.__vector

    @vector.setter
    def vector(self, vector: np.ndarray):
        self.__vector = copy.deepcopy(vector)

    @property
    def neighbors(self):
        return self.__neighbors

    @neighbors.setter
    def neighbors(self, neighbors: set):
        self.__neighbors = copy.deepcopy(neighbors)


def convert_graph_to_networkx(vertices: dict) -> nx.classes.graph.Graph:
    '''
    Convert graph to networkx graph type

    Parameters
    ----------
    vertices: dict

    Returns
    -------
    networkx_graph : nx.classes.graph.Graph
    '''
    networkx_graph = nx.Graph()
    graph = Graph(vertices)
    edges = graph.get_weighted_edges()
    networkx_graph.add_weighted_edges_from(edges)
    return networkx_graph

def nx_weighted_diameter_and_radius(graph: nx.classes.graph.Graph) -> tuple:
    '''
    Calculate weighted diameter and radius from NetworkX graph

    Parameters
    ----------
    graph: nx.classes.graph.Graph

    Returns
    -------
    diameter_radius : tuple
    '''
    diameter = 0
    radius = float('inf')
    file_ = open('stream_nx_weighted_diameter_and_radius.txt', 'w')
    for word in tqdm(graph.nodes, file=file_):
        word_paths_length = nx.single_source_dijkstra_path_length(graph, word)
        max_word = max(word_paths_length, key=lambda word: word_paths_length[word])
        max_length = word_paths_length[max_word]
        if max_length > diameter:
            diameter = max_length
            file_.write(f"\n word: {word}, max_word: {max_word}, dist_to_max_word: {max_length}, new_diameter: {diameter} \n")
        if max_length < radius:
            radius = max_length
            file_.write(f"\n word: {word}, max_word: {max_word}, dist_to_max_word: {max_length}, new_radius: {radius} \n")
    file_.close()
    return diameter, radius

def go_through_text(text: str, graph: nx.classes.graph.Graph) -> list:
    '''
    Convert text to path on NetworkX graph

    Parameters
    ----------
    text: str
    graph: nx.classes.graph.Graph

    Returns
    -------
    path_edges : list

    text = "word_1 word_2 ... word_n"
    Shortest path from word_i to word_i+1 is nx.dijkstra_path(graph, word_i, word_i+1)
    Shortest paths through text [(word_1, word_1_1, word_1_2, ..., word_1_n, word_2),
                                 (word_2, ..., word_3),
                                 (word_3, ..., word_4) ..., (word_n-1, ..., word_n)]
    '''
    text = text.split()
    path_edges = []
    current_word = text[0]
    file_ = open("stream_go_through_text.txt", 'w')
    for word in tqdm(text[1:], file=file_):
        if current_word not in graph:
            current_word = word
            continue
        if word not in graph:
            continue
        if word in graph[current_word]:
            path_edges.append((current_word, word))
            file_.write('in path_edges!!!')
        else:
            try:
                shortest_path = tuple(nx.dijkstra_path(graph, current_word, word))
                path_edges.append(shortest_path)
            except:
                path_edges.append(f"No path {current_word} -- {word}")
        current_word = word
    file_.close()
    return path_edges

def convert_text_shortest_paths_to_text_edges_path(shortest_paths: list) -> list:
    '''
    Convert text shortest paths to text edges path

    Parameters
    ----------
    shortest_paths: list

    Returns
    -------
    edges_path : list

    shortest_paths = [(word_1, word_1_1, word_1_2, ..., word_1_n, word_2),
                      (word_2, ..., word_3),
                      (word_3, ..., word_4) ..., (word_n-1, ..., word_n)]
    edges_path = [(word_1, word_1_1),
                  (word_1_2, word_1_3),
                   ...,
                  (word_n-1, word_n-1_1),
                   ...,
                  (word_n-1_k, word_n)]
    '''
    edges_path = []
    for short_path in shortest_paths:
        current_word = short_path[0]
        for word in short_path[1:]:
            edges_path.append((current_word, word))
            current_word = word
    return edges_path

def convert_text_shortest_paths_to_text_words_path(shortest_paths: list) -> list:
    '''
    Convert text shortest paths to text words path

    Parameters
    ----------
    shortest_paths: list

    Returns
    -------
    words_path : list

    shortest_paths = [(word_1, word_1_1, word_1_2, ..., word_1_n, word_2),
                      (word_2, ..., word_3),
                      (word_3, ..., word_4) ..., (word_n-1, ..., word_n)]
    words_path = [word_1, word_1_1, word_1_2, word_1_3,
                   ...,
                  word_n-1, word_n-1_1,
                   ...,
                  word_n-1_k, word_n]
    '''
    words_path = []
    for word in shortest_paths[0]:
        words_path.append(word)
    for short_path in shortest_paths[1:]:
        for word in short_path[1:]:
            words_path.append(word)
    return words_path

def convert_edges_path_to_words_path(edges_path: list) -> list:
    '''
    Convert edges path to words path

    Parameters
    ----------
    edges_path: list

    Returns
    -------
    words_path : list

    edges_path = [(word_1, word_1_1),
                  (word_1_2, word_1_3),
                   ...,
                  (word_n-1, word_n-1_1),
                   ...,
                  (word_n-1_k, word_n)]
    words_path = [word_1, word_1_1, word_1_2, word_1_3,
                   ...,
                  word_n-1, word_n-1_1,
                   ...,
                  word_n-1_k, word_n]
    '''
    words_path = []
    words_path.append(edges_path[0][0])
    for edge in edges_path:
        words_path.append(edge[1])
    return words_path


class Graph:
    def __init__(self, vertices: dict):
        '''
        Graph initialization

        Parameters
        ----------
        vertices: dict
        vertices = {word: Node(vector,
                               neighbors set)}
        '''
        self.__vertices = copy.deepcopy(vertices)

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self, vertices: dict):
        self.__vertices = copy.deepcopy(vertices)

    def get_words(self) -> list:
        return list(self.vertices.keys())

    def get_vectors(self) -> list:
        vectors = list()
        for node in self.vertices.values():
            vectors.append(node.vector.tolist())
        return vectors

    def get_edges(self) -> set:
        '''
        Get edges

        Returns
        -------
        edges : set
        edges = ((word, neighbor), ...)
        '''
        edges = set()
        for word, node in self.vertices.items():
            for neighbor in node.neighbors:
                if (word, neighbor) in edges or (neighbor, word) in edges:
                    continue
                edges.add((word, neighbor))
        return edges

    def get_weighted_edges(self) -> list:
        '''
        Get weighted edges

        Returns
        -------
        edges : set
        edges = ((word, neighbor, weight), ...)
        '''
        edges = self.get_edges()
        weighted_edges = list()
        for edge in tqdm(edges):
            word, neighbor = edge
            weight = self.euclid_distance(word, neighbor)
            weighted_edges.append((word, neighbor, weight))
        return weighted_edges

    def get_other_words(self, *words) -> set:
        '''
        Get other words

        Parameters
        ----------
        words : word_1, word_2, ...

        Returns
        -------
        other_words : list
        other_words = all_words - words
        '''
        return set(self.get_words()).difference(set(words))

    def get_graph_degrees(self) -> list:
        '''
        Get graph degrees

        Returns
        -------
        graph_degrees : list
        '''
        graph_degrees = []
        for word in self.get_words():
            graph_degrees.append(len(self.vertices[word].neighbors))
        return graph_degrees

    def add_edge(self, first_word: str, second_word: str):
        '''
        Add edge to word

        Parameters
        ----------
        first_word : str,
        second_word: str
        '''
        self.vertices[first_word].neighbors.add(second_word)
        self.vertices[second_word].neighbors.add(first_word)

    def delete_edge(self, first_word: str, second_word: str):
        '''
        Delete edge

        Parameters
        ----------
        first_word : str,
        second_word: str
        '''
        self.vertices[first_word].neighbors.difference_update(set([second_word]))
        self.vertices[second_word].neighbors.difference_update(set([first_word]))

    def reset_graph_neighbors(self):
        '''
        Reset graph neighbors
        '''
        for node in self.vertices.values():
            node.neighbors.clear()

    def euclid_distance(self, first_word: str, second_word: str) -> np.float64:
        '''
        Calculate Euclid distance from first_word to second_word

        Parameters
        ----------
        first_word: str,
        second_word: str

        Returns
        -------
        euclid_distance : np.float64
        '''
        first_vertex = self.vertices[first_word].vector
        second_vertex = self.vertices[second_word].vector
        return np.linalg.norm(first_vertex - second_vertex)

    def euclid_distance_to_vector(self, word: str, vector: np.ndarray) -> np.float64:
        '''
        Calculate Euclid distance from word to vector

        Parameters
        ----------
        word: str,
        vector: np.ndarray

        Returns
        -------
        euclid_distance : np.float64
        '''
        word_vector = self.vertices[word].vector
        return np.linalg.norm(word_vector - vector)

    def get_nearest_neighbor(self, word: str) -> str:
        '''
        Get nearest neighbor to word

        Parameters
        ----------
        word: str

        Returns
        -------
        nearest_neighbor : str
        '''
        neighbor_distance = float('inf')
        neighbor = ''
        for other_word in self.get_other_words(word):
            if self.euclid_distance(word, other_word) < neighbor_distance:
                neighbor_distance = self.euclid_distance(word, other_word)
                neighbor = other_word
        return neighbor

    def get_sphere_radius(self, word: str) -> np.float64:
        '''
        Get sphere radius for word

        Parameters
        ----------
        word: str

        Returns
        -------
        sphere_radius : np.float64
        '''
        nearest_neighbor = self.get_nearest_neighbor(word)
        return self.euclid_distance(word, nearest_neighbor)

    def get_vertices_and_distance(self, word: str) -> dict:
        '''
        Get vertices and Euclid distance to words

        Parameters
        ----------
        word: str

        Returns
        -------
        vertices_and_distance : dict
        '''
        vertices_and_distance = dict()
        for other_word in self.get_other_words(word):
            vertices_and_distance[other_word] = self.euclid_distance(word, other_word)
        return vertices_and_distance

    def get_knn(self, word: str, k: int) -> list:
        '''
        Get k-nearest neighbors

        Parameters
        ----------
        word: str
        k : int

        Returns
        -------
        k-nearest neighbors : list
        '''
        distance_dict = dict()
        for other_word in self.get_other_words(word):
            distance_dict[other_word] = self.euclid_distance(word, other_word)
        lambda_ = lambda item: item[1]
        words = [key for key, value in sorted(distance_dict.items(), key=lambda_)]
        return words[:k]

    def in_sphere_check(self, first_word: str, second_word: str) -> bool:
        '''
        Check if second_word in first_word's sphere

        Parameters
        ----------
        first_word : str
        second_word : str

        Returns
        -------
        answer : bool
        '''
        nearest_neighbor = self.get_knn(first_word, 1)
        radius = get_sphere_radius(first_word)
        distance = self.euclid_distance(first_word, second_word)
        if distance > radius:
            return False
        return True

    def update_graph_by_document_neighbors(self, text_neighbors: dict):
        '''
        Update graph by document neighbors, document neighbors interception

        Parameters
        ----------
        text_neighbors: dict
        '''
        for word in tqdm(self.vertices.keys()):
            self.vertices[word].neighbors.intersection_update(text_neighbors[word].document_neighbors)

    def update_graph_by_sentence_neighbors(self, text_neighbors: dict):
        '''
        Update graph by sentence neighbors, sentence neighbors interception

        Parameters
        ----------
        text_neighbors: dict
        '''
        for word in tqdm(self.vertices.keys()):
            self.vertices[word].neighbors.intersection_update(text_neighbors[word].sentence_neighbors)

    def degree_distribution(self) -> list:
        '''
        Get graph degrees distribution

        Returns
        -------
        degree_distribution : list
        '''
        degree_distribution = list()
        for word in self.get_words():
            degree_distribution.append(len(self.vertices[word].neighbors))
        return degree_distribution

    def get_eccentricity(self) -> dict:
        '''
        Get graph eccentricity

        Returns
        -------
        eccentricity : dict
        '''
        graph = nx.Graph()
        graph.add_weighted_edges_from(self.get_weighted_edges())
        eccentricity = dict()
        file_ = open('stream_eccentricity.txt', 'w')
        for word in tqdm(graph.nodes, file=file_):
            short_paths_length, short_paths = nx.single_source_dijkstra(graph, word)
            short_paths_full = dict()
            for word in short_paths.keys():
                short_paths_full[word] = tuple([short_paths[word], short_paths_length[word]])
            eccentricity[word] = short_paths_full
        file_.close()
        return eccentricity

    def get_eccentricity_for_words(self, words: set) -> dict:
        '''
        Get graph eccentricity for words

        Parameters
        ----------
        words: set

        Returns
        -------
        eccentricity_for_words : dict
        '''
        graph = nx.Graph()
        graph.add_weighted_edges_from(self.get_weighted_edges())
        eccentricity = dict()
        file_ = open('stream_eccentricity.txt', 'w')
        for word in tqdm(words, file=file_):
            if word not in graph.nodes:
                continue
            short_paths_length, short_paths = nx.single_source_dijkstra(graph, word)
            short_paths_full = dict()
            for other_word in short_paths.keys():
                short_paths_full[other_word] = tuple([short_paths[other_word], short_paths_length[other_word]])
            eccentricity[word] = short_paths_full
        file_.close()
        return eccentricity

    def diameter(self) -> float:
        '''
        Calculate graph diametr

        Returns
        -------
        diameter : float
        '''
        diameter = -1
        file_ = open('diameter_progress.txt', 'w')
        words = self.get_words()
        for word in tqdm(words, file=file_):
            short_paths = self.dijkstra(word)
            for path in short_paths:
                if diameter < path[0]:
                    diameter = path[0]
            file_.write(str(diameter) + '\n')
            file_.close()
        return diameter

    def diameter_for_words(self, words: set) -> float:
        '''
        Calculate graph diametr for words

        Parameters
        ----------
        words: set

        Returns
        -------
        diameter : float
        '''
        diameter = -1
        file_ = open('diameter_progress.txt', 'w')
        for word in tqdm(words, file=file_):
            short_paths = self.dijkstra(word)
            for path in short_paths:
                if diameter < path[0]:
                    diameter = path[0]
            file_.write(str(diameter) + '\n')
            file_.close()
        return diameter

    def get_networkx_graph() -> nx.classes.graph.Graph:
        '''
        Convert graph to NetworkX graph type

        Returns
        -------
        networkx_graph : nx.classes.graph.Graph
        '''
        networkx_graph = nx.Graph()
        edges = self.get_weighted_edges()
        networkx_graph.add_weighted_edges_from(edges)
        return networkx_graph

    def get_text_subgraph(self, text: list):
        '''
        Get text subgraph

        Parameters
        ----------
        text: list

        Returns
        -------
        networkx_subgraph : nx.classes.graph.Graph
        '''
        networkx_graph = self.get_networkx_graph()
        return networkx_graph.subgraph(text)

    def dijkstra(self, word: str) -> list:
        '''
        Calculate dijkstra shortest path from word to other words

        Parameters
        ----------
        word: str

        Returns
        -------
        min_dist_list : list
        '''
        words_list = self.get_words()
        word_node_index = words_list.index(word)
        dijkstra_nodes = [DijkstraNodeDecorator(other_word, index, self.vertices[other_word]) for index, other_word in enumerate(words_list)]
        dijkstra_nodes[word_node_index].provisional_distance = 0
        dijkstra_nodes[word_node_index].hops.append(self.vertices[word])
        is_less_than = lambda dijkstra_node_1, dijkstra_node_2: dijkstra_node_1.provisional_distance < dijkstra_node_2.provisional_distance
        get_index = lambda dijkstra_node: dijkstra_node.index()
        heap = MinHeap(dijkstra_nodes, is_less_than, get_index)
        min_dist_list = []
        # file_ = open('dijkstra_progress.txt', 'w')
        while heap.get_word_nodes_size() > 0:
            # file_.write(str(heap.get_word_nodes_size()))
            print(heap.get_word_nodes_size())
            # file_.write('\n')
            # Получает узел кучи, что еще не просматривался ('seen')
            # и находится на минимальном расстоянии от исходного узла
            min_decorated_node = heap.pop()
            min_dist = min_decorated_node.provisional_distance
            hops = min_decorated_node.hops
            min_dist_list.append([min_dist, hops])
            # Получает все следующие перескоки. Это больше не O(n^2) операция
            current_word = min_decorated_node.word
            word_neighbors = self.vertices[current_word].neighbors
            connections = list()
            for neighbor in word_neighbors:
                neighbor_index = words_list.index(neighbor)
                edge_weight = self.euclid_distance(current_word, neighbor)
                connections.append((neighbor_index, edge_weight))
            # Для каждой связи обновляет ее путь и полное расстояние от
            # исходного узла, если общее расстояние меньше, чем текущее расстояние
            # в массиве dist
            for (neighbor_index, edge_weight) in connections:
                word = words_list[neighbor_index]
                node = self.vertices[word]
                heap_location = heap.order_mapping[neighbor_index]
                if (heap_location is not None):
                    tot_dist = edge_weight + min_dist
                    if tot_dist < heap.dijkstra_nodes[heap_location].provisional_distance:
                        hops_cpy = list(hops)
                        hops_cpy.append(node)
                        data = {'provisional_distance': tot_dist, 'hops': hops_cpy}
                        heap.decrease_key(heap_location, data)
        # file_.close()
        return min_dist_list


class DijkstraNodeDecorator:
    def __init__(self, word: str, word_index: int, node: Node):
        self.word = word
        self.index = word_index
        self.node = node
        self.provisional_distance = float('inf')
        self.hops = []

    def data(self):
        return self.node

    def update_data(self, data):
        self.provisional_distance = data['provisional_distance']
        self.hops = data['hops']


class WordsBinaryTree:
    def __init__(self, dijkstra_nodes: list = list()):
        self.dijkstra_nodes = dijkstra_nodes

    def get_word_node_index(self, word: int) -> int:
        return self.dijkstra_nodes.index(word)

    def get_word_node_at_index(self, index: int) -> str:
        return self.dijkstra_nodes[index]

    def get_parent_index(self, index: int) -> int:
        return (index - 1) // 2

    def get_left_child_index(self, index: int) -> int:
        return 2 * index + 1

    def get_right_child_index(self, index: str) -> int:
        return 2 * index + 2

    def get_parent_word_node(self, word: str):
        return self.get_word_node_at_index(self.get_parent_index(word))

    def get_left_child_word_node(self, word: str):
        return self.get_word_node_at_index(self.get_left_index(word))

    def get_right_child_word_node(self, word: str):
        return self.get_word_node_at_index(self.get_right_index(word))

    def get_word_nodes_size(self):
        return len(self.dijkstra_nodes)


class MinHeap(WordsBinaryTree):
    def __init__(self,
                 dijkstra_nodes: list,
                 is_less_than = lambda a, b: a.provisional_distance < b.provisional_distance,
                 get_index = None):
        WordsBinaryTree.__init__(self, dijkstra_nodes)
        self.order_mapping = list(range(len(dijkstra_nodes)))
        self.is_less_than, self.get_index = is_less_than, get_index
        self.min_heapify()

    def min_heapify(self):
        for node_index in range(len(self.dijkstra_nodes), -1, -1):
            self.min_heapify_subtree(node_index)

    def min_heapify_subtree(self, node_index: int):
        size = self.get_word_nodes_size()
        left_index = self.get_left_child_index(node_index)
        right_index = self.get_right_child_index(node_index)
        minimum_index = node_index

        if left_index < size and self.is_less_than(self.dijkstra_nodes[left_index], self.dijkstra_nodes[minimum_index]):
            minimum_index = left_index
        if right_index < size and self.is_less_than(self.dijkstra_nodes[right_index], self.dijkstra_nodes[minimum_index]):
            minimum_index = right_index
        if minimum_index != node_index:
            min_node = self.dijkstra_nodes[minimum_index]
            self.dijkstra_nodes[minimum_index] = self.dijkstra_nodes[node_index]
            self.dijkstra_nodes[node_index] = min_node
            if self.get_index is not None:
                self.order_mapping[self.dijkstra_nodes[minimum_index].index] = minimum_index
                self.order_mapping[self.dijkstra_nodes[node_index].index] = node_index
            self.min_heapify_subtree(minimum_index)

    def get_minimum_node(self):
        return self.dijkstra_nodes[0]

    def pop(self):
        min_node = self.get_minimum_node()
        if self.get_word_nodes_size() > 1:
            self.dijkstra_nodes[0] = self.dijkstra_nodes[-1]
            self.dijkstra_nodes.pop()
            # Обновляет order_mapping, если можно
            if self.get_index is not None:
                # print(self.dijkstra_nodes[0].index())
                self.order_mapping[self.dijkstra_nodes[0].index] = 0
            self.min_heapify_subtree(0)
        elif self.get_word_nodes_size() == 1:
            self.dijkstra_nodes.pop()
        else:
            return None
        # Если self.get_index существует, обновляет self.order_mapping для указания, что
        # узел индекса больше не в куче
        if self.get_index is not None:
            # Устанавливает значение None для self.order_mapping для обозначения непринадлежности к куче
            self.order_mapping[min_node.index] = None
        return min_node

    # Обновляет значение узла и подстраивает его, если нужно, чтобы сохранить свойства кучи
    def decrease_key(self, index, data):
        self.dijkstra_nodes[index].update_data(data)
        parent_index = self.get_parent_index(index)
        while index != 0 and not self.is_less_than(self.dijkstra_nodes[parent_index], self.dijkstra_nodes[index]):
            parent_node = self.dijkstra_nodes[parent_index]
            self.dijkstra_nodes[parent_index] = self.dijkstra_nodes[index]
            self.dijkstra_nodes[index] = parent_node
            # Если есть лямбда для получения индекса узла
            # обновляет массив order_mapping для указания, где именно находится индекс
            # в массиве узлов (осмотр этого индекса O(1))
            if self.get_index is not None:
                self.order_mapping[self.dijkstra_nodes[parent_index].index] = parent_index
                self.order_mapping[self.dijkstra_nodes[index].index] = index
            index = parent_index
            parent_index = self.get_parent_index(index) if index > 0 else None



'''
Параллелизация графов
'''


@ray.remote
class Parallel(Graph):
    def __init__(self, vertices: dict):
        '''
        Parallel graph initialization

        Parameters
        ----------
        vertices: dict
        vertices = {word: Node(vector,
                               neighbors set)}
        '''
        super().__init__(vertices)
        self.__parallel = set()

    @property
    def parallel(self):
        return self.__parallel

    @parallel.setter
    def parallel(self, parallel: set):
        self.__parallel = copy.deepcopy(parallel)

    def add_parallel_word(self, word: str):
        '''
        Add parallel word

        Parameters
        ----------
        word: str
        '''
        self.parallel.add(word)

    def add_parallel_triangle(self, triangle: set):
        '''
        Add parallel triangle

        Parameters
        ----------
        triangle: set
        '''
        self.parallel.add(tuple(triangle))

    def parallel_eball_graph(self, epsilon: float, index: int) -> dict:
        '''
        Create parallel process to calculate e-ball subgraph

        Parameters
        ----------
        epsilon: float
        index: int

        Returns
        -------
        subgraph_vertices : dict
        '''
        file_ = open(f"eball_progress_{index}.txt", 'w')
        for first_word in tqdm(self.parallel, file=file_):
            other_words = self.get_other_words(first_word)
            for second_word in other_words:
                distance = self.euclid_distance(first_word, second_word)
                if distance < epsilon:
                    self.add_edge(first_word, second_word)
        file_.close()
        return self.vertices

    def is_neighbor_in_edge_sphere_gromov(self, edge: tuple, neighbor: str) -> bool:
        '''
        Check if neighbor in edge sphere, formula by Gromov

        Parameters
        ----------
        edge : tuple
        neighbor : str

        Returns
        -------
        answer : bool
        '''
        edge_length = self.euclid_distance(edge[0], edge[1])
        first_to_neighbor = self.euclid_distance(edge[0], neighbor)
        second_to_neighbor = self.euclid_distance(edge[1], neighbor)
        if edge_length < math.sqrt(first_to_neighbor ** 2 + second_to_neighbor ** 2):
            return False
        return True

    def is_neighbor_in_edge_sphere(self, edge: tuple, neighbor: str) -> bool:
        '''
        Check if neighbor in edge sphere

        Parameters
        ----------
        edge : tuple
        neighbor : str

        Returns
        -------
        answer : bool
        '''
        sphere_center = (self.vertices[edge[0]].vector + self.vertices[edge[1]].vector) / 2
        radius = self.euclid_distance_to_vector(edge[0], sphere_center)
        if self.euclid_distance_to_vector(neighbor, sphere_center) < radius:
            return True
        return False

    def parallel_gabriel_graph(self, index: int) -> dict:
        '''
        Create parallel process to calculate gabriel subgraph

        Parameters
        ----------
        index: int

        Returns
        -------
        vertices_subgraph : dict
        '''
        file_ = open(f"gabriel_progress_{index}.txt", 'w')
        for triangle in tqdm(self.parallel, file=file_):
            edges = list(itertools.combinations(triangle, 2))
            for edge in edges:
                neighbors = set(triangle).difference(set(edge))
                for neighbor in neighbors:
                    if self.is_neighbor_in_edge_sphere(edge, neighbor):
                        self.delete_edge(edge[0], edge[1])
        file_.close()
        return self.vertices

    def parallel_rn_graph(self, index: int) -> dict:
        '''
        Create parallel process to calculate relative neighborhood subgraph

        Parameters
        ----------
        index: int

        Returns
        -------
        vertices_subgraph : dict
        '''
        file_ = open(f"rn_progress_{index}.txt", 'w')
        for word in tqdm(self.parallel, file=file_):
            neighbors = copy.deepcopy(self.vertices[word].neighbors)
            for neighbor in neighbors:
                for other_word in self.get_other_words(word, neighbor):
                    word_to_neighbor_distance = self.euclid_distance(word, neighbor)
                    word_to_other_word_distance = self.euclid_distance(word, other_word)
                    neighbor_to_other_word_distance = self.euclid_distance(neighbor, other_word)
                    if max(word_to_other_word_distance, neighbor_to_other_word_distance) <= word_to_neighbor_distance:
                        self.delete_edge(word, neighbor)
        file_.close()
        return self.vertices

    def parallel_influence_graph(self, index: int) -> dict:
        '''
        Create parallel process to calculate influence subgraph

        Parameters
        ----------
        index: int

        Returns
        -------
        vertices_subgraph : dict
        '''
        file_ = open(f"influence_progress_{index}.txt", 'w')
        for first_word in tqdm(self.parallel, file=file_):
            first_radius = self.get_sphere_radius(first_word)
            for second_word in self.get_other_words(first_word):
                second_radius = self.get_sphere_radius(second_word)
                distance = self.euclid_distance(first_word, second_word)
                if distance <= first_radius + second_radius:
                    self.add_edge(first_word, second_word)
        file_.close()
        return self.vertices

    def parallel_nn_graph(self, k: int, index: int) -> dict:
        '''
        Create parallel process to calculate k-nearest neighbors subgraph

        Parameters
        ----------
        index: int

        Returns
        -------
        vertices_subgraph : dict
        '''
        file_ = open(f"nn_progress_{index}.txt", 'w')
        for word in tqdm(self.parallel, file=file_):
            knn_list = self.get_knn(word, k)
            for neighbor in knn_list:
                self.add_edge(word, neighbor)
        file_.close()
        return self.vertices


'''
Триангуляция Делоне (DT) и Граф Габриэля (GG)
'''


class GG(Graph):
    def __init__(self,
                 vertices: dict,
                 triangles: list = list(),
                 delaunay: scipy.spatial.qhull.Delaunay = None):
                '''
                Gabriel graph initialization

                Parameters
                ----------
                vertices: dict
                vertices = {word: Node(vector,
                                       neighbors set)}
                triangles: list = list()
                delaunay: scipy.spatial.qhull.Delaunay = None
                '''
        super().__init__(vertices)
        self.__triangles = copy.deepcopy(triangles)
        self.__delaunay = copy.deepcopy(delaunay)

    @property
    def triangles(self):
        return self.__triangles

    @triangles.setter
    def triangles(self, triangles: list):
        self.__triangles = copy.deepcopy(triangles)

    @property
    def delaunay(self):
        return self.__delaunay

    @delaunay.setter
    def delaunay(self, delaunay: scipy.spatial.qhull.Delaunay):
        self.__delaunay = copy.deepcopy(delaunay)

    def create_delaunay_graph(self):
        '''
        Create Delaunay triangulation
        '''
        self.reset_graph_neighbors()
        vectors = self.get_vectors()
        words = self.get_words()
        word_num_dict = {word: num for word, num in enumerate(words)}
        file_ = open('delaunay_progress.txt', 'w')
        file_.write("Delaunay start... \n")
        self.delaunay = Delaunay(np.array(vectors), qhull_options="Qbb Qc Qz Qx Q12")
        delaunay_graph = self.delaunay.simplices.tolist()
        file_.write("Delaunay done!")
        for triangle in tqdm(delaunay_graph, file=file_):
            triangle_words = set(map(word_num_dict.get, triangle))
            self.triangles.append(triangle_words)
            for word in triangle_words:
                new_neighbors = triangle_words.difference(set([word]))
                self.vertices[word].neighbors.update(new_neighbors)
        file_.close()

    def is_neighbor_in_edge_sphere_gromov(self, edge: tuple, neighbor: str) -> bool:
        '''
        Check if neighbor in edge sphere, formula by Gromov

        Parameters
        ----------
        edge : tuple
        neighbor : str

        Returns
        -------
        answer : bool
        '''
        edge_length = self.euclid_distance(edge[0], edge[1])
        first_to_neighbor = self.euclid_distance(edge[0], neighbor)
        second_to_neighbor = self.euclid_distance(edge[1], neighbor)
        if edge_length < math.sqrt(first_to_neighbor ** 2 + second_to_neighbor ** 2):
            return False
        return True

    def is_neighbor_in_edge_sphere(self, edge: tuple, neighbor: str) -> bool:
        '''
        Check if neighbor in edge sphere

        Parameters
        ----------
        edge : tuple
        neighbor : str

        Returns
        -------
        answer : bool
        '''
        sphere_center = (self.vertices[edge[0]].vector + self.vertices[edge[1]].vector) / 2
        radius = self.euclid_distance_to_vector(edge[0], sphere_center)
        if self.euclid_distance_to_vector(neighbor, sphere_center) < radius:
            return True
        return False

    def create_gabriel_graph(self):
        '''
        Create Gabriel graph
        '''
        self.create_delaunay_graph()
        file_ = open('gabriel_progress.txt', 'w')
        for triangle in tqdm(self.triangles, file=file_):
            edges = list(itertools.combinations(triangle, 2))
            for edge in edges:
                neighbors = triangle.difference(set(edge))
                for neighbor in neighbors:
                    if self.is_neighbor_in_edge_sphere(edge, neighbor):
                        self.delete_edge(edge[0], edge[1])
        file_.close()

    def create_gabriel_graph_for_words(self, words: set) -> dict:
        '''
        Create Gabriel subgraph for words

        Parameters
        ----------
        words: set

        Returns
        -------
        vertices_subgraph : dict
        '''
        file_ = open('stream_gg_for_words.txt', 'w')
        for word in tqdm(words, file=file_):
            for possible_neighbor in self.get_other_words(word):
                neighbor_flag = True
                for other_word in self.get_other_words(word, possible_neighbor):
                    possible_edge = (word, possible_neighbor)
                    if self.is_neighbor_in_edge_sphere(possible_edge, other_word):
                        neighbor_flag = False
                        break
                if neighbor_flag:
                    self.vertices[word].neighbors.update(possible_neighbor)
        file_.write("Gabriel updating...")
        vertices_with_neighbors = dict()
        for word in words:
            vertices_with_neighbors[word] = self.vertices[word]
        file_.write("GNG done!")
        file_.close()
        return vertices_with_neighbors

    def create_parallel_gabriel_graph(self, triangles: list, num_cpus: int = 1):
        '''
        Create parallel Gabriel graph

        Parameters
        ----------
        triangles : list
        num_cpus : int = 1
        '''
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        file_ = open('stream_create_progress.txt', 'w')
        for index, triangle in enumerate(tqdm(triangles, file=file_)):
            streaming_actors[index % num_cpus].add_parallel_triangle.remote(triangle)
        file_.write("\n Ready for ray get \n")
        file_.close()
        results = ray.get([
            actor.parallel_gabriel_graph.remote(index) for index, actor in enumerate(streaming_actors)
        ])
        ray.shutdown()
        file_ = open('merge_streams_progress.txt', 'w')
        for parallel_vertices in tqdm(results, file=file_):
            for word in self.get_words():
                self.vertices[word].neighbors.update(parallel_vertices[word].neighbors)
        file_.close()


'''
Граф относительного соседства (RNG)
'''


class RNG(GG):
    def __init__(self,
                 vertices: dict,
                 triangles: list = list(),
                 delaunay: scipy.spatial.qhull.Delaunay = None):
                 '''
                 Relative Neighborhood Graph (RNG) initialization

                 Parameters
                 ----------
                 vertices : dict
                 triangles : list = list()
                 delaunay : scipy.spatial.qhull.Delaunay = None
                 '''
        super().__init__(vertices, triangles, delaunay)

    def create_rn_graph(self, already_gabriel: bool = True):
        '''
        Create RNG

        Parameters
        ----------
        already_gabriel: bool = True
        '''
        if not already_gabriel:
            self.create_gabriel_graph()
        file_ = open('stream_rng.txt', 'w')
        for word in tqdm(self.get_words(), file=file_):
            neighbors = copy.deepcopy(self.vertices[word].neighbors)
            for neighbor in neighbors:
                for other_word in self.get_other_words(word, neighbor):
                    word_to_neighbor_distance = self.euclid_distance(word, neighbor)
                    word_to_other_word_distance = self.euclid_distance(word, other_word)
                    neighbor_to_other_word_distance = self.euclid_distance(neighbor, other_word)
                    if max(word_to_other_word_distance, neighbor_to_other_word_distance) <= word_to_neighbor_distance:
                        self.delete_edge(word, neighbor)
        file_.close()

    def create_rn_graph_for_words(self, words: set, already_gabriel: bool = True) -> dict:
        '''
        Create RNG subgraph for words

        Parameters
        ----------
        words: set
        already_gabriel: bool = True

        Returns
        -------
        vertices_subgraph : dict
        '''
        if not already_gabriel:
            self.create_gabriel_graph()
        file_ = open('stream_rng_for_words.txt', 'w')
        for word in tqdm(words, file=file_):
            neighbors = copy.deepcopy(self.vertices[word].neighbors)
            for neighbor in neighbors:
                for other_word in self.get_other_words(word, neighbor):
                    word_to_neighbor_distance = self.euclid_distance(word, neighbor)
                    word_to_other_word_distance = self.euclid_distance(word, other_word)
                    neighbor_to_other_word_distance = self.euclid_distance(neighbor, other_word)
                    if max(word_to_other_word_distance, neighbor_to_other_word_distance) <= word_to_neighbor_distance:
                        self.delete_edge(word, neighbor)
        file_.write("RNG updating...")
        vertices_with_neighbors = dict()
        for word in words:
            vertices_with_neighbors[word] = self.vertices[word]
        file_.write("RNG done!")
        file_.close()
        return vertices_with_neighbors

    def create_parallel_rn_graph(self, num_cpus: int = 1, use_input_as_gabriel: bool = True):
        '''
        Create parallel RNG

        Parameters
        ----------
        num_cpus : int = 1
        use_input_as_gabriel : bool = True
        '''
        if not use_input_as_gabriel:
            file_ = open('gabriel_progress.txt', 'w')
            file_.write("Gabriel start... \n")
            self.create_parallel_gabriel_graph()
            file_.write("Gabriel done! \n")
            file_.close()
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        file_ = open('stream_create_progress.txt', 'w')
        for index, word in enumerate(tqdm(self.vertices, file=file_)):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        file_.close()
        results = ray.get([
            actor.parallel_rn_graph.remote(index) for index, actor in enumerate(streaming_actors)
        ])
        ray.shutdown()
        file_ = open('merge_streams_progress.txt', 'w')
        for parallel_vertices in tqdm(results, file=file_):
            for word in self.get_words():
                self.vertices[word].neighbors.intersection_update(parallel_vertices[word].neighbors)
        file_.close()


'''
Граф влияния (IG)
'''


class IG(Graph):
    def __init__(self, vertices: dict):
        '''
        Influence Graph (IG) initialization

        Parameters
        ----------
        vertices : dict
        '''
        super().__init__(vertices)

    def get_full_sphere_radius(self) -> dict:
        '''
        Get words sphere radius

        Returns
        -------
        words_sphere_radius : dict
        '''
        words_sphere_radius = dict()
        file_ = open('stream_get_words_sphere_radius.txt', 'w')
        for word in tqdm(self.get_words(), file=file_):
            words_sphere_radius[word] = self.get_sphere_radius(word)
        file_.close()
        return words_sphere_radius

    def get_sphere_radius_for_words(self, words: set) -> dict:
        '''
        Get sphere radius for words

        Parameters
        ----------
        words: set

        Returns
        -------
        words_sphere_radius : dict
        '''
        words_sphere_radius = dict()
        file_ = open('stream_get_sphere_radius_for_words.txt', 'w')
        for word in tqdm(words, file=file_):
            words_sphere_radius[word] = self.get_sphere_radius(word)
        file_.close()
        return words_sphere_radius

    def create_influence_graph(self):
        '''
        Create IG
        '''
        self.reset_graph_neighbors()
        for first_word in tqdm(self.vertices.keys()):
            other_words = self.get_other_words(first_word)
            first_radius = self.get_sphere_radius(first_word)
            for second_word in other_words:
                second_radius = self.get_sphere_radius(second_word)
                distance = self.euclid_distance(first_word, second_word)
                if distance <= first_radius + second_radius:
                    self.add_edge(first_word, second_word)

    def create_influence_graph_for_words(self, words: set, sphere_radius: dict) -> dict:
        '''
        Create IG for words

        Parameters
        ----------
        words : set
        sphere_radius : dict

        Returns
        -------
        vertices_with_neighbors : dict
        '''
        self.reset_graph_neighbors()
        file_ = open('stream_create_influence_graph_for_words.txt', 'w')
        for first_word in tqdm(words, file=file_):
            for second_word in self.get_other_words(first_word):
                distance = self.euclid_distance(first_word, second_word)
                if distance <= sphere_radius[first_word] + sphere_radius[second_word]:
                    self.add_edge(first_word, second_word)
        file_.write("IG updating...")
        vertices_with_neighbors = dict()
        for word in words:
            vertices_with_neighbors[word] = self.vertices[word]
        file_.write("IG done!")
        file_.close()
        return vertices_with_neighbors

    def create_parallel_influence_graph(self):
        '''
        Create parallel IG for words
        '''
        self.reset_graph_neighbors()
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        for index, word in enumerate(self.vertices):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        results = ray.get([
            actor.parallel_influence_graph.remote() for actor in streaming_actors
        ])
        ray.shutdown()
        for parallel_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors.update(parallel_vertices[word].neighbors)


'''
Граф ε-окружности (ε-ball)
'''


class EBall(Graph):
    def __init__(self, vertices: dict):
        '''
        Epsilon Ball Graph (EBall) initialization

        Parameters
        ----------
        vertices : dict
        '''
        super().__init__(vertices)

    def create_eball_graph(self, epsilon: float):
        '''
        Create Eball graph

        Parameters
        ----------
        epsilon : float
        '''
        self.reset_graph_neighbors()
        for first_word in self.get_words():
            other_words = self.get_other_words(first_word)
            for second_word in other_words:
                distance = self.euclid_distance(first_word, second_word)
                if distance < epsilon:
                    self.add_edge(first_word, second_word)

    def create_eball_graph_for_words(self, words: set, epsilon: float) -> dict:
        '''
        Create EBall for words

        Parameters
        ----------
        words : set
        epsilon: float

        Returns
        -------
        vertices_with_neighbors : dict
        '''
        self.reset_graph_neighbors()
        file_ = open("progress.txt", 'w')
        for first_word in tqdm(words, file=file_):
            other_words = self.get_other_words(first_word)
            for second_word in other_words:
                distance = self.euclid_distance(first_word, second_word)
                if distance < epsilon:
                    self.add_edge(first_word, second_word)
        file_.write("Eball updating...")
        vertices_with_neighbors = dict()
        for word in words:
            vertices_with_neighbors[word] = self.vertices[word]
        file_.write("Eball done!")
        file_.close()
        return vertices_with_neighbors

    def create_parallel_eball_graph(self, epsilon: float, num_cpus: int):
        '''
        Create parallel EBall graph

        Parameters
        ----------
        epsilon : float
        num_cpus : int
        '''
        self.reset_graph_neighbors()
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        file_ = open('stream_create_progress.txt', 'w')
        for index, word in enumerate(tqdm(self.get_words(), file=file_)):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        file_.close()
        pool = ActorPool(streaming_actors)
        indexes = [actor_index for actor_index in range(len(streaming_actors))]
        results = pool.map(lambda actor, index: actor.parallel_eball_graph.remote(epsilon, index), indexes)
        ray.shutdown()
        file_ = open('merge_progress.txt', 'w')
        for parallel_vertices in tqdm(results, file=file_):
            for word in parallel_vertices.parallel:
                self.vertices[word].neighbors.update(parallel_vertices[word].neighbors)
        file_.close()


'''
Граф k-ближайших соседей (NNG)
'''


class NNG(Graph):
    def __init__(self, vertices: dict):
        '''
        K-Nearest Neighbor Graph initialization

        Parameters
        ----------
        vertices : dict
        '''
        super().__init__(vertices)

    def create_nn_graph(self, k: int):
        '''
        Create KNN graph

        Parameters
        ----------
        k : int
        '''
        self.reset_graph_neighbors()
        for word in tqdm(self.vertices.keys()):
            knn_list = self.get_knn(word, k)
            for neighbor in knn_list:
                self.add_edge(word, neighbor)

    def create_parallel_nn_graph(self, k: int):
        '''
        Create parallel KNN graph

        Parameters
        ----------
        k : int
        '''
        self.reset_graph_neighbors()
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus)
        streaming_actors = [Parallel.remote(self.vertices) for _ in range(num_cpus)]
        for index, word in enumerate(self.vertices):
            streaming_actors[index % num_cpus].add_parallel_word.remote(word)
        results = ray.get([
            actor.parallel_nn_graph.remote(k) for actor in streaming_actors
        ])
        ray.shutdown()
        for parallel_vertices in results:
            for word in self.get_words():
                self.vertices[word].neighbors.update(parallel_vertices[word].neighbors)
