# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:25:25 2019

@author: songl
"""
import itertools
import pickle
import random
import unittest

import networkx

from explorable_graph import ExplorableGraph
from submission import a_star, bidirectional_a_star, \
    bidirectional_ucs, breadth_first_search, euclidean_dist_heuristic, \
    null_heuristic, haversine_dist_heuristic, tridirectional_search, tridirectional_upgraded, \
    uniform_cost_search, custom_heuristic

with open('romania_graph.pickle', 'rb') as rom:
    romania = pickle.load(rom)
romania = ExplorableGraph(romania)
path=tridirectional_search(romania,['a','s','o'])
