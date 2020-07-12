import json
import math
import re
import csv
import os
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, Birch, KMeans
from sklearn.model_selection import train_test_split
from networkx.algorithms import approximation as approx
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, calinski_harabaz_score, calinski_harabasz_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

TOTAL_AUTHOR = 4085
P_L = 25000
N_L = 100000

class DataFrame:
    def __init__(self, raw_data, tr_ratio=0.9):
        self.rd = raw_data
        self.edges = raw_data.train_edges
        self.pred_edges = raw_data.pred_edges
        self.key_features = raw_data.key_features
        self.tr_ratio = tr_ratio

        self.d = {'CN': 1,
                  'Salton': 1,
                  'Sorenson': 1,
                  'HPI': 1,
                  'HDI': 1,
                  'LHN': 1,
                  'AA': 1,
                  'RA': 1,
                  'D': 1,
                  'NC': 1,
                  'KIDF': 1,
                  'Katz': 1,
                  'VIDF': 1,
                  'NP': 1,
                  'Y': 1
                  }

        self._process()

    def _process(self):
        self.fe = FeatureExtractor(self.key_features)
        self._process_edges()
        self._extract_all_feature()

    def _process_edges(self):
        edges = self.edges
        G = self._generate_graph(edges)
        G_init = G.copy()
        self.init_graph = G_init
        # initial_components = nx.number_connected_components(G)
        #
        # try:
        #     omissible_edges = np.load('OLI.npy')
        # except:
        #     omissible_edges = []
        #     edges = list(G.edges)
        #     for u, v in tqdm(edges, desc='Searching omissible edges'):
        #         G.remove_edge(u, v)
        #         if nx.number_connected_components(G) == initial_components:
        #             omissible_edges.append((u, v))
        #         else:
        #             G.add_edge(u, v)
        #     omissible_edges = np.array(omissible_edges)
        #     np.save('OLI.npy', omissible_edges)
        # omissible_edges = [(edge[0], edge[1]) for edge in omissible_edges]
        # print('omissible edges', len(omissible_edges))
        # hub_edges = [edge for edge in edges if edge not in omissible_edges]
        # print('hub edges', len(hub_edges))
        # hub_G = self._generate_graph(hub_edges)

        true_edges = self.get_true_edges()
        fake_edges = self.get_fake_edges()
        print('true edges', len(true_edges))
        print('fake_edges', len(fake_edges))

        # train_dev_edges, reple_edges = self._cut_edges(omissible_edges, number=P_L)
        # train_G = hub_G.copy()
        # self._add_edges(train_G, reple_edges)

        train_G = self.init_graph
        train_dev_edges = true_edges


        self.train_graph, self.train_dev_edges, self.fake_edges = train_G, train_dev_edges, fake_edges

    def get_true_edges(self):
        graph = self.init_graph
        adj_matrix = nx.to_numpy_matrix(graph)
        adj_matrix = np.triu(adj_matrix, 1)
        true_edge_index_x,  true_edge_index_y = np.where(adj_matrix == 1)
        true_edge_index = []
        for i in range(len(true_edge_index_x)):
            true_edge_index.append((true_edge_index_x[i], true_edge_index_y[i]))
        print('shuffling')
        random.shuffle(true_edge_index)
        node_list = list(graph.nodes)

        true_edge_index = true_edge_index[:P_L]
        true_edges = []
        self.true_edges_distribution = defaultdict(int)
        for i in tqdm(true_edge_index):

            u, v, = node_list[i[0]], node_list[i[1]]
            if (u, v,) not in self.pred_edges:
                d = self.fe.get_D_index(u, v, graph)
                true_edges.append((u, v,))
                self.true_edges_distribution[d]+=1

        print(self.true_edges_distribution)

        return true_edges


    def get_fake_edges(self):
        graph = self.init_graph
        adj_matrix = nx.to_numpy_matrix(graph)
        r_adj_matrix = np.triu(np.where(adj_matrix == 0, 2, 0), 1)
        fake_edge_index_x, fake_edge_index_y = np.where(r_adj_matrix == 2)
        fake_edge_index = []
        for i in range(len(fake_edge_index_y)):
            fake_edge_index.append((fake_edge_index_x[i], fake_edge_index_y[i]))
        print('shuffling')
        random.shuffle(fake_edge_index)
        print('shuffle finished')
        self.calulate_fake_edge_distribute()

        sp_d = self.fake_edge_dis

        node_list = list(graph.nodes)
        # unconnected_edges = []
        # c_sp_d = defaultdict(int)
        # number = N_L
        # print({k: int(v*number) for k, v in sp_d.items()})
        # for i in fake_edge_index:
        #     u, v, = node_list[i[0]], node_list[i[1]]
        #     d = min(self.fe.get_D_index(u, v, self.init_graph), 7)
        #     if d != 0 and int(sp_d[d] * number) > c_sp_d[d] and (u, v,) not in self.pred_edges:
        #         c_sp_d[d] += 1
        #         unconnected_edges.append((u, v,))
        #
        #         print({k: int(v * number) for k, v in sp_d.items()})
        #         print(c_sp_d)
        #
        #     if sum(c_sp_d.values()) % (number // 10) == 0:
        #         perc = sum(c_sp_d.values()) // (number // 100)
        #         print('Extracting Negative Edges: [' + perc // 5 * '=' + '>' + '.' * (20 - perc // 5) + '] ' + str(
        #             perc) + '% finished')
        #     if sum(c_sp_d.values()) == sum({k: int(v*number) for k, v in sp_d.items()}.values()):
        #         break

        fake_edge_index = fake_edge_index[:N_L]
        unconnected_edges = []
        self.fake_edge_distribution  = defaultdict(list)
        for i in tqdm(fake_edge_index):

            u, v, = node_list[i[0]], node_list[i[1]]
            if (u, v,) not in self.pred_edges:
                d = self.fe.get_D_index(u, v, graph)
                unconnected_edges.append((u, v,))
                self.fake_edge_distribution[d].append((u, v))

        print([(k, len(v)) for k, v in self.fake_edge_distribution.items()])

        fake_edges = []

        min_ratio = 1
        for d, edges in self.fake_edge_distribution.items():
            if self.true_edges_distribution[d] != 0:
                ratio = len(edges)/ self.true_edges_distribution[d]
                if ratio<min_ratio:
                    min_ratio = ratio


        for d, edges in self.fake_edge_distribution.items():
            if self.true_edges_distribution[d] != 0:
                random.shuffle(edges)
                n = int(self.true_edges_distribution[d]*min_ratio)
                fake_edges.extend(edges[:n])

        print('fake_edges', len(fake_edges))

        return fake_edges



    def calulate_fake_edge_distribute(self):
        dis_dict = defaultdict(int)
        true_edges = self.edges
        for u, v in true_edges:
            d = self.fe.get_D_index(u, v, self.init_graph)
            if d != 0:
                dis_dict[d] += 1
        # print(dis_dict)
        edge_count = sum(dis_dict.values())

        pred_dis_dict = defaultdict(int)
        for u, v in self.pred_edges:
            d = self.fe.get_D_index(u, v, self.init_graph)
            if d != 0:
                pred_dis_dict[d] += 1

        edge_dis_count = {k: v / edge_count * 1000 for k, v in dis_dict.items()}
        nega_edge_dis_count = {k: (pred_dis_dict[k] - edge_dis_count[k])/1000  for k in dis_dict}
        self.fake_edge_dis = nega_edge_dis_count
        self.fake_edge_dis.pop(10)


    def _add_edges(self, graph, edges):
        for u, v in edges:
            graph.add_edge(u, v)

    def _generate_graph(self, edges):
        G = nx.Graph()
        for i in range(TOTAL_AUTHOR):
            G.add_node(str(i))
        for u, v in edges:
            G.add_edge(u, v)
        return G

    def _cut_edges(self, edges, ratio=0.0, number=0):
        indexes = [i for i in range(len(edges))]
        random.shuffle(indexes)

        if ratio != 0:
            tr_len = int(len(edges) * ratio)
        else:
            tr_len = number
        tr_indexes = indexes[:tr_len]
        te_indexes = indexes[tr_len:]

        tr_edges = [edges[i] for i in tr_indexes]
        te_edges = [edges[i] for i in te_indexes]

        return tr_edges, te_edges

    def _extract_all_feature(self):
        graph, train_dev_edges, fake_edges, pred_edges = self.train_graph, self.train_dev_edges, self.fake_edges, self.pred_edges

        train_edges, dev_edges = self._cut_edges(train_dev_edges, ratio=0.8)
        train_fake_edges, dev_fake_edges = self._cut_edges(fake_edges, ratio=0.8)

        self.fe.set_katz_matrix(graph, 0.1)
        print('train edges', len(train_edges))
        print('dev edges', len(dev_edges))

        train_graph = graph
        train_X, train_Y = self.get_data(train_graph, train_edges, train_fake_edges)
        clf = ExtraTreesClassifier()
        clf = clf.fit(train_X, train_Y)
        model = SelectFromModel(clf, prefit=True)
        print(model.get_support())
        train_X = model.transform(train_X)
        print('train data', train_X.shape, train_Y.shape)


        dev_graph = train_graph.copy()
        self._add_edges(dev_graph, dev_edges)
        dev_X, dev_Y = self.get_data(dev_graph, dev_edges, dev_fake_edges)
        dev_X = model.transform(dev_X)
        print(dev_X.shape, dev_Y.shape)


        pred_graph = dev_graph.copy()
        self._add_edges(pred_graph, pred_edges)
        pred_X, _ = self.get_data(pred_graph, pred_edges, [])
        pred_X = model.transform(pred_X)
        print(pred_X.shape)

        train_len = len(train_X)
        dev_len = len(dev_X)

        X = np.vstack([train_X, dev_X, pred_X])
        X = self.normalize(X)

        self.tr_x = X[:train_len]
        self.tr_y = train_Y
        self.te_x = X[train_len:train_len+dev_len]
        self.te_y = dev_Y
        self.pr_x = X[train_len+dev_len:]





    def normalize(self, data):
        _range = np.max(data, axis=0) - np.min(data, axis=0)
        return (data - np.min(data, axis=0)) / _range

    def get_data(self, graph, posi_edges, nega_edges, ):
        import copy
        pairs = copy.deepcopy(posi_edges)
        pairs.extend(nega_edges)
        train_features = []
        kwargs = self.d
        for k, v in kwargs.items():
            if v == 1:
                arr = []
                for u, v in tqdm(pairs, desc='Extracting ' + k + ' array'):
                    value = eval('self.fe.get_' + k + '_index')(u, v, graph)
                    arr.append(value)
                arr = [[f] for f in arr]
                train_features.append(arr)
        DATA_X = np.hstack(train_features)
        posi_y = np.array([1 for _ in range(len(posi_edges))])
        nega_y = np.array([0 for _ in range(len(nega_edges))])
        DATA_Y = np.r_[posi_y, nega_y]
        return DATA_X, DATA_Y


class FeatureExtractor():
    def __init__(self, key_features):
        self.key_features = key_features
        self._process_tf_idf()

    def _process_tf_idf(self):
        import re
        key_features = self.key_features
        number_corpus = len(key_features)

        author_dict = {}

        idf_count_dict = defaultdict(int)
        for fs in key_features:
            keys = fs.keys()
            for k in keys:
                idf_count_dict[k] += 1
            author_dict[str(fs['id'])] = fs

        key_idf_dict = {k: math.log(number_corpus / v) for k, v in idf_count_dict.items() if re.match('keyword', k)}
        venue_idf_dict = {k: math.log(number_corpus / v) for k, v in idf_count_dict.items() if re.match('venue', k)}

        self.key_idf_dict = key_idf_dict
        self.venue_idf_dict = venue_idf_dict
        self.author_dict = author_dict

        self._keys = list(self.key_idf_dict.keys())
        self._venues = list(self.venue_idf_dict.keys())

    def get_CN_index(self, u, v, graph):
        return len(list(nx.common_neighbors(graph, u, v)))

    def get_Salton_index(self, u, v, graph):
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = len(list(nx.common_neighbors(graph, u, v))) / ((nx.degree(graph, v)) * nx.degree(graph, u)) ** (
                    1 / 2) if len(list(nx.common_neighbors(graph, u, v))) != 0 else 0
            graph.add_edge(u, v)
        else:
            value = len(list(nx.common_neighbors(graph, u, v))) / ((nx.degree(graph, v)) * nx.degree(graph, u)) ** (
                    1 / 2) if len(list(nx.common_neighbors(graph, u, v))) != 0 else 0
        return value

    def get_Sorenson_index(self, u, v, graph):
        # return np.array([2 * len(CN_dict[edge]) / (D_dict[edge[0]] + D_dict[edge[1]]) for edge in edges])
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = len(list(nx.common_neighbors(graph, u, v))) / (
                    (nx.degree(graph, v)) + nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
            graph.add_edge(u, v)
        else:
            value = len(list(nx.common_neighbors(graph, u, v))) / (
                    (nx.degree(graph, v)) + nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
        return value

    def get_HPI_index(self, u, v, graph):
        # return np.array([2 * len(CN_dict[edge]) / (D_dict[edge[0]] + D_dict[edge[1]]) for edge in edges])
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = len(list(nx.common_neighbors(graph, u, v))) / min((nx.degree(graph, v)),
                                                                      nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
            graph.add_edge(u, v)
        else:
            value = len(list(nx.common_neighbors(graph, u, v))) / min((nx.degree(graph, v)),
                                                                      nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
        return value

    def get_HDI_index(self, u, v, graph):
        # return np.array([2 * len(CN_dict[edge]) / (D_dict[edge[0]] + D_dict[edge[1]]) for edge in edges])
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = len(list(nx.common_neighbors(graph, u, v))) / max((nx.degree(graph, v)),
                                                                      nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
            graph.add_edge(u, v)
        else:
            value = len(list(nx.common_neighbors(graph, u, v))) / max((nx.degree(graph, v)),
                                                                      nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
        return value

    def get_LHN_index(self, u, v, graph):
        # return np.array([2 * len(CN_dict[edge]) / (D_dict[edge[0]] + D_dict[edge[1]]) for edge in edges])
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = len(list(nx.common_neighbors(graph, u, v))) / (
                    (nx.degree(graph, v)) * nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
            graph.add_edge(u, v)
        else:
            value = len(list(nx.common_neighbors(graph, u, v))) / (
                    (nx.degree(graph, v)) * nx.degree(graph, u)) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
        return value

    def get_AA_index(self, u, v, graph):
        # return np.array([2 * len(CN_dict[edge]) / (D_dict[edge[0]] + D_dict[edge[1]]) for edge in edges])
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = sum(1 / math.log(nx.degree(graph, n)) for n in list(nx.common_neighbors(graph, u, v))) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
            graph.add_edge(u, v)
        else:
            value = sum(1 / math.log(nx.degree(graph, n)) for n in list(nx.common_neighbors(graph, u, v))) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
        return value

    def get_RA_index(self, u, v, graph):
        # return np.array([2 * len(CN_dict[edge]) / (D_dict[edge[0]] + D_dict[edge[1]]) for edge in edges])
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = sum(1 / nx.degree(graph, n) for n in list(nx.common_neighbors(graph, u, v))) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
            graph.add_edge(u, v)
        else:
            value = sum(1 / nx.degree(graph, n) for n in list(nx.common_neighbors(graph, u, v))) if len(
                list(nx.common_neighbors(graph, u, v))) != 0 else 0
        return value

    def get_D_index(self, u, v, graph):
        # return np.array([2 * len(CN_dict[edge]) / (D_dict[edge[0]] + D_dict[edge[1]]) for edge in edges])
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            try:
                value = nx.shortest_path_length(graph, u, v, )
            except:
                value = 10
            graph.add_edge(u, v)
        else:
            try:
                value = nx.shortest_path_length(graph, u, v, )
            except:
                value = 10
        if u == v:
            value = 0
        return value


    def get_NC_index(self, u, v, graph):
        if (u, v) in nx.edges(graph):
            graph.remove_edge(u, v)
            value = approx.local_node_connectivity(graph, u, v, ) if u != v else -1
            graph.add_edge(u, v)
        else:
            value = approx.local_node_connectivity(graph, u, v, ) if u != v else -1
        return value

    def get_KIDF_index(self, u, v, graph):

        _keys = self._keys
        u_dict = self.author_dict[u]
        v_dict = self.author_dict[v]

        u_vec = [self.key_idf_dict[k] if u_dict.get(k) is not None else 0 for k in _keys]
        v_vec = [self.key_idf_dict[k] if v_dict.get(k) is not None else 0 for k in _keys]
        if np.linalg.norm(u_vec) * np.linalg.norm(v_vec) == 0:
            value = 0
        else:
            value = np.dot(u_vec, v_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(v_vec))
        return value

    def set_katz_matrix(self, graph, beta):
        A = nx.to_numpy_matrix(graph)
        size = list(graph.nodes).__len__()
        I = np.eye(size)
        node_list = list(graph.nodes)
        S = np.linalg.inv(I-beta*A)-I
        self.S = S
        self.node_list = node_list


    def get_VIDF_index(self, u, v, graph):

        _venues = self._venues
        u_dict = self.author_dict[u]
        v_dict = self.author_dict[v]

        u_vec = [self.venue_idf_dict[k] if u_dict.get(k) is not None else 0 for k in _venues]
        v_vec = [self.venue_idf_dict[k] if v_dict.get(k) is not None else 0 for k in _venues]
        if np.linalg.norm(u_vec) * np.linalg.norm(v_vec) == 0:
            value = 0
        else:
            value = np.dot(u_vec, v_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(v_vec))
        return value

    def get_NP_index(self, u, v, graph):
        u_n = self.author_dict[u]['num_papers']
        v_n = self.author_dict[v]['num_papers']
        return abs(u_n - v_n)

    def get_Y_index(self, u, v, graph):
        u_f, u_l = self.author_dict[u]['first'], self.author_dict[u]['last']
        v_f, v_l = self.author_dict[v]['first'], self.author_dict[v]['last']
        return min(u_f, v_f) - max(u_l, v_l)

    def get_Katz_index(self, u, v, graph):
        i = self.node_list.index(u)
        j = self.node_list.index(v)
        value = self.S[i, j]
        return value