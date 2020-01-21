# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:54:18 2019

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
    def test_bidirectional_a_star(self):
        """Test and generate GeoJSON for bidirectional A* search"""
        path = bidirectional_a_star(self.atlanta, '69581003', '69581000', heuristic=haversine_dist_heuristic)
        all_explored = self.atlanta.explored_nodes
        plot_search(self.atlanta, 'atlanta_search_bidir_a_star.json', path,
                    all_explored)

if __name__ == '__main__':
    unittest.main()