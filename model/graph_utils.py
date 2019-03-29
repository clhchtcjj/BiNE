__author__ = 'CLH'

import networkx as nx
import graph
import random
from networkx.algorithms import bipartite as bi
import numpy as np
from lsh import get_negs_by_lsh
from io import open
import os
import itertools

class GraphUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.G = nx.Graph()
        self.edge_dict_u = {}
        self.edge_dict_v = {}
        self.edge_list = []
        self.node_u = []
        self.node_v = []
        self.authority_u, self.authority_v = {}, {}
        self.walks_u, self.walks_v = [], []
        self.G_u, self.G_v = None, None
        self.fw_u = os.path.join(self.model_path, "homogeneous_u.dat")
        self.fw_v = os.path.join(self.model_path, "homogeneous_v.dat")
        self.negs_u = {}
        self.negs_v = {}
        self.context_u = {}
        self.context_v = {}

    def construct_training_graph(self, filename=None):
        if filename is None:
            filename = os.path.join(self.model_path, "ratings_train.dat")
        edge_list_u_v = []
        edge_list_v_u = []
        with open(filename, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                if self.edge_dict_u.get(user) is None:
                    self.edge_dict_u[user] = {}
                if self.edge_dict_v.get(item) is None:
                    self.edge_dict_v[item] = {}
                edge_list_u_v.append((user, item, float(rating)))
                self.edge_dict_u[user][item] = float(rating)
                self.edge_dict_v[item][user] = float(rating)
                edge_list_v_u.append((item, user, float(rating)))
                line = fin.readline()
        # create bipartite graph
        self.node_u = self.edge_dict_u.keys()
        self.node_v = self.edge_dict_v.keys()
        self.node_u.sort()
        self.node_v.sort()
        self.G.add_nodes_from(self.node_u, bipartite=0)
        self.G.add_nodes_from(self.node_v, bipartite=1)
        self.G.add_weighted_edges_from(edge_list_u_v+edge_list_v_u)
        self.edge_list = edge_list_u_v

    def calculate_centrality(self, mode='hits'):
        if mode == 'degree_centrality':
            a = nx.degree_centrality(self.G)
        else:
            h, a = nx.hits(self.G)

        max_a_u, min_a_u,max_a_v,min_a_v = 0, 100000, 0, 100000

        for node in self.G.nodes():
            if node[0] == "u":
                if max_a_u < a[node]:
                    max_a_u = a[node]
                if min_a_u > a[node]:
                    min_a_u = a[node]
            if node[0] == "i":
                if max_a_v < a[node]:
                    max_a_v = a[node]
                if min_a_v > a[node]:
                    min_a_v = a[node]

        for node in self.G.nodes():
            if node[0] == "u":
                if max_a_u-min_a_u != 0:
                    self.authority_u[node] = (float(a[node])-min_a_u) / (max_a_u-min_a_u)
                else:
                    self.authority_u[node] = 0
            if node[0] == 'i':
                if max_a_v-min_a_v != 0:
                    self.authority_v[node] = (float(a[node])-min_a_v) / (max_a_v-min_a_v)
                else:
                    self.authority_v[node] = 0

    def homogeneous_graph_random_walks(self, percentage, maxT, minT):
        # print(len(self.node_u),len(self.node_v))
        A = bi.biadjacency_matrix(self.G, self.node_u, self.node_v, dtype=np.float,weight='weight', format='csr')
        row_index = dict(zip(self.node_u, itertools.count()))
        col_index = dict(zip(self.node_v, itertools.count()))
        index_row = dict(zip(row_index.values(), row_index.keys()))
        index_item = dict(zip(col_index.values(), col_index.keys()))
        AT = A.transpose()
        self.save_homogenous_graph_to_file(A.dot(AT),self.fw_u, index_row,index_row)
        self.save_homogenous_graph_to_file(AT.dot(A),self.fw_v, index_item,index_item)
        self.G_u, self.walks_u = self.get_random_walks_restart(self.fw_u, self.authority_u, percentage=percentage, maxT=maxT, minT=minT)
        self.G_v, self.walks_v = self.get_random_walks_restart(self.fw_v, self.authority_v, percentage=percentage, maxT=maxT, minT=minT)


    def get_random_walks_restart(self, datafile, hits_dict, percentage, maxT, minT):
        if datafile is None:
            datafile = os.path.join(self.model_path,"rating_train.dat")
        G = graph.load_edgelist(datafile, undirected=True)
        print("number of nodes: {}".format(len(G.nodes())))
        print("walking...")
        walks = graph.build_deepwalk_corpus_random(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0)
        print("walking...ok")
        return G, walks

    def homogeneous_graph_random_walks_for_large_bipartite_graph(self, percentage, maxT, minT):
        A = bi.biadjacency_matrix(self.G, self.node_u, self.node_v, dtype=np.float,weight='weight', format='csr')
        row_index = dict(zip(self.node_u, itertools.count()))
        col_index = dict(zip(self.node_v, itertools.count()))
        index_row = dict(zip(row_index.values(), row_index.keys()))
        index_item = dict(zip(col_index.values(), col_index.keys()))
        AT = A.transpose()
        matrix_u = self.get_homogenous_graph(A.dot(AT), self.fw_u, index_row, index_row)
        matrix_v = self.get_homogenous_graph(AT.dot(A), self.fw_v, index_item, index_item)
        self.G_u, self.walks_u = self.get_random_walks_restart_for_large_bipartite_graph(matrix_u, self.authority_u, percentage=percentage, maxT=maxT, minT=minT)
        self.G_v, self.walks_v = self.get_random_walks_restart_for_large_bipartite_graph(matrix_v, self.authority_v, percentage=percentage, maxT=maxT, minT=minT)

    def homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(self, datafile, percentage, maxT, minT):
        self.G_u, self.walks_u = self.get_random_walks_restart_for_large_bipartite_graph_without_generating(datafile, self.authority_u, percentage=percentage, maxT=maxT, minT=minT, node_type='u')
        self.G_v, self.walks_v = self.get_random_walks_restart_for_large_bipartite_graph_without_generating(datafile, self.authority_v, percentage=percentage, maxT=maxT, minT=minT,node_type='i')

    def get_random_walks_restart_for_large_bipartite_graph(self, matrix, hits_dict, percentage, maxT, minT):
        G = graph.load_edgelist_from_matrix(matrix, undirected=True)
        print("number of nodes: {}".format(len(G.nodes())))
        print("walking...")
        walks = graph.build_deepwalk_corpus_random(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0)
        print("walking...ok")
        return G, walks

    def get_random_walks_restart_for_large_bipartite_graph_without_generating(self, datafile, hits_dict, percentage, maxT, minT, node_type='u'):
        if datafile is None:
            datafile = os.path.join(self.model_path,"rating_train.dat")
        G = graph.load_edgelist(datafile, undirected=True)
        cnt = 0
        for n in G.nodes():
            if n[0] == node_type:
                cnt += 1
        print("number of nodes: {}".format(cnt))
        print("walking...")
        walks = graph.build_deepwalk_corpus_random_for_large_bibartite_graph(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0,node_type=node_type)
        # print(walks)
        print("walking...ok")
        return G, walks


    def save_words_and_sentences_to_file(self, filenodes, filesentences):
        with open(filenodes,"w") as fw:
            for node in self.G.keys():
                fw.write(node+"\n")

        with open(filesentences,"w") as fs:
            for nodes in self.walks:
                for index in range(0,len(nodes)):
                    if index == len(nodes)-1:
                        fs.write(nodes[index]+"\n")
                    else:
                        fs.write(nodes[index]+" ")
    def get_negs(self,num_negs):
        self.negs_u, self.negs_v = get_negs_by_lsh(self.edge_dict_u,self.edge_dict_v,num_negs)
        # print(len(self.negs_u),len(self.negs_v))
        return self.negs_u, self.negs_v

    def get_context_and_fnegatives(self,G,walks,win_size,num_negs,table):
        # generate context and negatives
        if isinstance(G, graph.Graph):
            node_list = G.nodes()
        elif isinstance(G, list):
            node_list = G
        word2id = {}
        for i in range(len(node_list)):
            word2id[node_list[i]] = i + 1
        walk_list = walks
        print("context...")
        context_dict = {}
        new_neg_dict = {}
        for step in range(len(walk_list)):

            walk = walk_list[step % len(walk_list)]
            # print(walk)
            batch_labels = []
            # travel each walk
            for iter in range(len(walk)):
                start = max(0, iter - win_size)
                end = min(len(walk), iter + win_size + 1)
                # index: index in window
                if context_dict.get(walk[iter]) is None:
                    context_dict[walk[iter]] = []
                    new_neg_dict[walk[iter]] = []
                labels_list = []
                neg_sample = []
                for index in range(start, end):
                    labels_list.append(walk[index])
                while len(neg_sample) < num_negs:
                    sa = random.choice(range(len(node_list)))
                    if table[sa] in labels_list:
                        continue
                    neg_sample.append(table[sa])
                context_dict[walk[iter]].append(labels_list)
                new_neg_dict[walk[iter]].append(neg_sample)
            if len(batch_labels) == 0:
                continue
        print("context...ok")
        return context_dict, new_neg_dict

    def get_context_and_negatives(self,G,walks,win_size,num_negs,negs_dict):
        # generate context and negatives
        if isinstance(G, graph.Graph):
            node_list = G.nodes()
        elif isinstance(G, list):
            node_list = G
        word2id = {}
        for i in range(len(node_list)):
            word2id[node_list[i]] = i + 1
        walk_list = walks
        print("context...")
        context_dict = {}
        new_neg_dict = {}
        for step in range(len(walk_list)):
            walk = walk_list[step % len(walk_list)]
            # print(walk)
            # travel each walk
            for iter in range(len(walk)):
                start = max(0, iter - win_size)
                end = min(len(walk), iter + win_size + 1)
                # index: index in window
                if context_dict.get(walk[iter]) is None:
                    context_dict[walk[iter]] = []
                    new_neg_dict[walk[iter]] = []
                labels_list = []
                negs = negs_dict[walk[iter]]
                for index in range(start, end):
                    if walk[index] in negs:
                        negs.remove(walk[index])
                    if walk[index] == walk[iter]:
                        continue
                    else:
                        labels_list.append(walk[index])
                neg_sample = random.sample(negs,min(num_negs,len(negs)))
                context_dict[walk[iter]].append(labels_list)
                new_neg_dict[walk[iter]].append(neg_sample)
        print("context...ok")
        return context_dict, new_neg_dict
        # with open(context_file,'w', encoding='utf-8') as fw1, open(neg_file,'w', encoding='utf-8') as fw2:
        #     for u in context_dict.keys():
        #         fw1.write(u+"\t")
        #         fw2.write(u+"\t")
        #         lens = len(context_dict[u])
        #         for i in range(lens):
        #             str1 = u','.join(context_dict[u][i])
        #             str2 = u','.join(neg_dict[u][i])
        #             if i != lens -1:
        #                 fw1.write(str1+"\t")
        #                 fw2.write(str2+"\t")
        #             else:
        #                 fw1.write(str1+"\n")
        #                 fw2.write(str2+"\n")
        # return context_dict, neg_dict

    def save_homogenous_graph_to_file(self, A, datafile, index_row, index_item):
        (M,N) = A.shape
        csr_dict = A.__dict__
        data = csr_dict.get("data")
        indptr = csr_dict.get("indptr")
        indices = csr_dict.get("indices")
        col_index = 0
        with open(datafile,'w') as fw:
            for row in range(M):
                for col in range(indptr[row],indptr[row+1]):
                    r = row
                    c = indices[col]
                    fw.write(index_row.get(r)+"\t"+index_item.get(c)+"\t"+str(data[col_index])+"\n")
                    col_index += 1

    def get_homogenous_graph(self, A, datafile, index_row, index_item):
        (M,N) = A.shape
        csr_dict = A.__dict__
        data = csr_dict.get("data")
        indptr = csr_dict.get("indptr")
        indices = csr_dict.get("indices")
        col_index = 0
        matrix = {}
        with open(datafile,'w') as fw:
            for row in range(M):
                for col in range(indptr[row],indptr[row+1]):
                    r = index_row.get(row)
                    c = index_item.get(indices[col])
                    if matrix.get(r) is None:
                        matrix[r] = []
                    matrix[r].append(c)
                    col_index += 1

        return matrix

    def read_sentences_and_homogeneous_graph(self, filesentences=None, datafile=None):
        G = graph.load_edgelist(datafile, undirected=True)
        walks = []
        with open(filesentences,"r") as fin:
            for line in fin.readlines():
                walk = line.strip().split(" ")
                walks.append(walk)
        return G, walks






