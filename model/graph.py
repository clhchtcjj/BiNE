#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count
import math
logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""
  act = {}
  isWeight = False
  def __init__(self):
    super(Graph, self).__init__(list)

  def setIsWeight(self,isWeight):
    self.isWeight = isWeight

  def initAct(self):
    for i in self.keys():
        self.act[i] = 0

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()

    if self.isWeight == True:
      for k in iterkeys(self):
        self[k] = self.sortedDictValues(self[k])
        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1-t0))
        self.remove_self_loops_dict()
    else:
      for k in iterkeys(self):
        self[k] = list(sorted(set(self[k])))
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self



  def sortedDictValues(self,adict):
    keys = adict.keys()
    keys.sort()
    return map(adict.get, keys)

  def make_consistent_dict(self):

    t0 = time()

    for k in iterkeys(self):
      self[k] = self.sortedDictValues(self[k])
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))
    self.remove_self_loops_dict()
    return self



  def remove_self_loops(self):

    removed = 0
    t0 = time()
    if self.isWeight == True:
      for x in self:
          if x in self[x].keys():
              del self[x][x]
              removed += 1
    else:
      for x in self:
        if x in self[x]:
          self[x].remove(x)
          removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()

  def random_walk(self, nodes, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(nodes)]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          add_node = rand.choice(G[rand.choice(G[cur])])
          while add_node == cur:
            add_node = rand.choice(G[rand.choice(G[cur])])
          path.append(add_node)
        else:
          path.append(path[0])
      else:
        break
    return path

  # def spreading_activation(self, path_length, alpha=0, rand=random.Random(), start=None):
  #   """ Returns a truncated random walk.
  #
  #       path_length: Length of the random walk.
  #       alpha: probability of restarts.
  #       start: the start node of the random walk.
  #   """
  #   G = self
  #   if start:
  #     path = [str(start)]
  #   else:
  #     # Sampling is uniform w.r.t V, and not w.r.t E
  #     path = [rand.choice(G.keys())]
  #
  #   while len(path) < path_length:
  #     cur = path[-1]
  #     if len(G[cur]) > 0:
  #       if rand.random() >= alpha:
  #         temp = rand.choice(G[cur])
  #         while
  #         path.append(rand.choice(G[cur]))
  #       else:
  #         path.append(path[0])
  #     else:
  #       break
  #   return path

  def random_walk_restart(self, nodes, percentage, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        percentage: probability of stopping walking
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(nodes)]

    while len(path) < 1 or random.random() > percentage:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          add_node = rand.choice(G[cur])
          while add_node == cur:
            add_node = rand.choice(G[cur])
          path.append(add_node)
        else:
          path.append(path[0])
      else:
        break
    return path
      # neighbors = []
      # for n in G[cur]:
      #   neighbors.extend(G[n])
      # if len(G[cur]) > 0:
      #   if rand.random() >= alpha:
      #     add_node = rand.choice(neighbors)
      #     path.append(add_node)
      #   else:
      #     path.append(path[0])
      # else:
      #   break
    # return path

  def random_walk_restart_for_large_bipartite_graph(self, nodes, percentage, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        percentage: probability of stopping walking
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(nodes)]
    while len(path) < 1 or random.random() > percentage:
      cur = path[-1]
      neighbors = set([])
      for nei in G[cur]:
          neighbors = neighbors.union(set(G[nei]))
      # print(len(neighbors))
      neighbors = list(neighbors)
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          add_node = rand.choice(neighbors)
          while add_node == cur and len(neighbors) > 1:
            add_node = rand.choice(neighbors)
          path.append(add_node)
        else:
          path.append(path[0])
      else:
        break
    return path

def calculateAct(self,node):
    G = self

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand = random.Random(), node_type='u'):
  walks = []
  
  nodes_total = list(G.nodes())
  nodes = []
  for obj in nodes_total:
    if obj[0] == node_type:
      nodes.append(obj)

  # nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
     walks.append(G.random_walk(nodes,path_length, alpha=alpha, rand=rand, start=node))
  
  return walks

def build_deepwalk_corpus_random(G, hits_dict, percentage, maxT, minT, alpha=0, rand = random.Random()):
  walks = []
  nodes = list(G.nodes())
  for node in nodes:
    num_paths = max(int(math.ceil(maxT * hits_dict[node])),minT)
    # print num_paths,
    for cnt in range(num_paths):
     walks.append(G.random_walk_restart(nodes, percentage,rand=rand, alpha=alpha, start=node))

  random.shuffle(walks)
  return walks

def build_deepwalk_corpus_random_for_large_bibartite_graph(G, hits_dict, percentage, maxT, minT, alpha=0, rand = random.Random(), node_type='u'):
  walks = []
  nodes_total = list(G.nodes())
  nodes = []
  # print(len(nodes),node_type)
  for obj in nodes_total:
    if obj[0] == node_type:
      nodes.append(obj)
  # cnt_0 = 1
  # print(len(nodes))
  for node in nodes:
    # if cnt_0 % 1000 == 0:
    #   print(cnt_0)
    # cnt_0 += 1
    num_paths = max(int(math.ceil(maxT * hits_dict[node])),minT)
    # print num_paths,
    for cnt in range(num_paths):
     walks.append(G.random_walk_restart_for_large_bipartite_graph(nodes, percentage,rand=rand, alpha=alpha, start=node))
  random.shuffle(walks)
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()

  with open(file_) as f:
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
      total = 0 
      for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
          adjlist.extend(adj_chunk)
          total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True):
  G = Graph()
  #adddddd
  with open(file_,encoding="UTF-8") as f:
    for l in f:
      x, y = l.strip().split()[:2]
      G[x].append(y)
      if undirected:
        G[y].append(x)
  G.make_consistent()
  return G

def load_edgelist_from_matrix(matrix, undirected=True):
  G = Graph()
  #adddddd
  for x in matrix.keys():
    for y in matrix[x]:
      G[x].append(y)
      if undirected:
        G[y].append(x)
  G.make_consistent()
  return G

def load_edgelist_w(file_, undirected=True):
  G = Graph()
  G.setIsWeight(True)
  G.initAct()
  with open(file_) as f:
    for l in f:
      x, y , w = l.strip().split()[:3]
      x = int(x)
      y = int(y)
      w = float(w)
      if len(G[x])==0:
          G[x] = {}
      if len(G[y])==0:
          G[y] = {}
      G[x][y] = w
      if undirected:
        G[y][x] = w

  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G


if __name__ == '__main__':
    G = Graph()