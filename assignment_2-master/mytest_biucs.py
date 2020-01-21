# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:24:04 2019

@author: songl
"""

# coding=utf-8
import pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from submission import PriorityQueue, a_star, bidirectional_a_star, \
    bidirectional_ucs, breadth_first_search, uniform_cost_search, haversine_dist_heuristic, \
    tridirectional_upgraded, custom_heuristic
from visualize_graph import plot_search

class TestBidirectionalSearch(unittest.TestCase):
    """Test the bidirectional search algorithms: UCS, A*"""

    def setUp(self):
        """Load Atlanta map data"""
        with open('atlanta_osm.pickle', 'rb') as atl:
            atlanta = pickle.load(atl)
        self.atlanta = ExplorableGraph(atlanta)
        self.atlanta.reset_search()

    def test_bidirectional_ucs(self):
        """Test and generate GeoJSON for bidirectional UCS search"""
        path = bidirectional_ucs(self.atlanta, '69581003', '69581000')
        all_explored = self.atlanta.explored_nodes
        plot_search(self.atlanta, 'atlanta_search_bidir_ucs.json', path,
                    all_explored)
        print(path)

if __name__ == '__main__':
    unittest.main()