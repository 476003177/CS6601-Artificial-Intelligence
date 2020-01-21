# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:26:39 2019

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

class TestBasicSearch(unittest.TestCase):
    def setUp(self):
        """Romania map data from Russell and Norvig, Chapter 3."""
        with open('romania_graph.pickle', 'rb') as rom:
            romania = pickle.load(rom)
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()
        
    def test_ucs(self):
        """TTest and visualize uniform-cost search"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.node[n]['pos'] for n in
                          self.romania.node.keys()}

        self.romania.reset_search()
        path = bidirectional_ucs(self.romania, start, goal)
        print(path)
        print(self.romania.explored_nodes)
        print(sum(1 for x in self.romania.explored_nodes.values() if x==1))
        
        self.romania.reset_search()
        path2 = uniform_cost_search(self.romania, start, goal)
        print(path2)
        print(self.romania.explored_nodes)
        print(sum(1 for x in self.romania.explored_nodes.values() if x==1))

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path)
    
    @staticmethod
    def draw_graph(graph, node_positions=None, start=None, goal=None,
                   path=None):
        """Visualize results of graph search"""
        explored = [key for key in graph.explored_nodes if graph.explored_nodes[key] > 0]

        labels = {}
        for node in graph:
            labels[node] = node

        if node_positions is None:
            node_positions = networkx.spring_layout(graph)

        networkx.draw_networkx_nodes(graph, node_positions)
        networkx.draw_networkx_edges(graph, node_positions, style='dashed')
        networkx.draw_networkx_labels(graph, node_positions, labels)

        networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored,
                                     node_color='g')
        edge_labels = networkx.get_edge_attributes(graph, 'weight')
        networkx.draw_networkx_edge_labels(graph, node_positions, edge_labels=edge_labels)
        
        if path is not None:
            edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
            networkx.draw_networkx_edges(graph, node_positions, edgelist=edges,
                                         edge_color='b')

        if start:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[start], node_color='b')

        if goal:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[goal], node_color='y')

        plt.plot()
        plt.show()
        
if __name__ == '__main__':
    unittest.main()