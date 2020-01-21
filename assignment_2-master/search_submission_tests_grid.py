# coding=utf-8
import pickle
import random
import unittest

import matplotlib.pyplot as plt
from matplotlib import pylab
import networkx as nx

#from explorable_graph_modified import ExplorableGraph
from explorable_graph import ExplorableGraph
from submission import tridirectional_search, bidirectional_ucs, tridirectional_upgraded, bidirectional_a_star

def save_graph(graph, file_name, show_edge_labels=False, show_node_labels=False, color_values=None):
    """
    Function saves graph onto disk using networkx libraries
    :param graph: networkx graph
    :param file_name: name for the png file
    :param show_edge_labels: flag to show edge labels
    :param show_node_labels: flag to show
    :param color_values: Map color values for nodes
    :return: None (Saves image file to disk)
    """
    # initialize Figure
    plt.figure(num=None, figsize=(50, 50), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spectral_layout(graph)
    if color_values is not None:
        nx.draw_networkx_nodes(graph, pos, node_color=color_values, cmap='plasma')
    else:
        nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    if show_node_labels:
        nx.draw_networkx_labels(graph, pos)

    #cut = 1.00
    #xmax = cut * max(xx for xx, yy in pos.values())
    #ymax = cut * max(yy for xx, yy in pos.values())
    #plt.xlim(0, xmax)
    #plt.ylim(0, ymax)

    plt.savefig(file_name)
    pylab.close()
    del fig


def create_grid(size=20, randomize_weights=False, file_name='grid_test.gpickle'):
    """
    Use this function to initialize a simple grid of given size and save it as a gpickle file.
    :param size: size of grid
    :param randomize_weights: flag to decide random weights for edges of the graph
    :param file_name: name of gpickle file
    :return: None (saves gpickle file to disk)
    """
    # Create simple grid
    grid = nx.grid_2d_graph(size, size)
    if randomize_weights:
        for s, t in grid.edges():
            grid[s][t]['weight'] = random.randint(1, 5)
    else:
        for s, t in grid.edges():
            grid[s][t]['weight'] = 1

    for n in grid.nodes():
        grid.node[n]['pos'] = n
        grid.node[n]['position'] = n

    nx.write_gpickle(grid, file_name)


class TestSearchExperimental(unittest.TestCase):
    """Test your search algorithms with a nxn grid based graph and visualize your results"""

    def setUp(self):
        """Load grid map data"""
        file_name = 'grid.gpickle'

        # Use this function to create any custom grid
        #create_grid(20, file_name=file_name)
        with open(file_name, "rb") as file:
            self.original_grid = pickle.load(file)

        self.grid = ExplorableGraph(self.original_grid)
        self.grid.reset_search()

    def test_grid_viz_bidirectional_search(self):
        """ Use this function to test out your code on a grid to visualize
        the paths and explored nodes for Bidirectional Search.
        This function will save the image files grid_expansion_bidirectional_search.png and
        grid_paths_bidirectional_search.png.
        """

        coordinates = [(0, 0), (6, 7)]
        path = bidirectional_ucs(self.grid, coordinates[0], coordinates[1])
        # path = bidirectional_a_star(self.grid, coordinates[0], coordinates[1], heuristic=custom_heuristic)
        explored = list(self.grid.explored_nodes.keys())

        """

        Color Map Code:
        * Nodes never explored : White
        * Nodes explored but not in path : Red
        * Nodes in path : Green

        """
        val_map = {
            0: {
                0: {0: 'w', 1: 'w'},
                1: {0: 'w', 1: 'w'},
            },
            1: {
                0: {0: 'r', 1: 'r'},
                1: {0: 'g', 1: 'b'}
            }
        }
        color_values = [val_map[node in explored][node in path][node in coordinates] for node in self.grid.nodes()]
        save_graph(self.original_grid, "grid_paths_bidirectional_search.png",
                   show_node_labels=True,
                   show_edge_labels=False,
                   color_values=color_values)

        expanded_nodes_dict = dict(self.grid.explored_nodes)
        # Color of nodes gets lighter as it gets explored more
        expansion_color_values = list(expanded_nodes_dict.values())
        save_graph(self.original_grid, "grid_expansion_bidirectional_search.png",
                   show_node_labels=True,
                   show_edge_labels=False,
                   color_values=expansion_color_values)

    def test_grid_viz_tridirectional_search(self):
        """ Use this function to test out your code on a grid to
        visualize the paths and explored nodes for Tridirectional Search.
        This function will save the image files grid_paths_tridirectional_search.png and
        grid_expansion_tridirectional_search.png
        """

        coordinates = [(0, 0), (5, 1), (19, 10)]
        path = tridirectional_search(self.grid, coordinates)
        # path = tridirectional_upgraded(self.grid, coordinates, heuristic=custom_heuristic)
        # path = three_bidirectional_search(self.grid, coordinates, heuristic=custom_heuristic)
        explored = list(self.grid.explored_nodes.keys())

        """

        Color Map Code:
        * Source/destination coordinates : Blue
        * Nodes never explored : White
        * Nodes explored but not in path : Red
        * Nodes in path : Green

        """
        val_map = {
            0: {
                0: {0: 'w', 1: 'w'},
                1: {0: 'w', 1: 'w'},
            },
            1: {
                0: {0: 'r', 1: 'r'},
                1: {0: 'g', 1: 'b'}
            }
        }
        color_values = [val_map[node in explored][node in path][node in coordinates] for node in self.grid.nodes()]
        save_graph(self.original_grid, "grid_paths_tridirectional_search.png",
                   show_node_labels=True,
                   show_edge_labels=False,
                   color_values=color_values)

        expanded_nodes_dict = dict(self.grid.explored_nodes)
        # Color of nodes gets lighter as it gets explored more
        expansion_color_values = list(expanded_nodes_dict.values())
        save_graph(self.original_grid, "grid_expansion_tridirectional_search.png",
                   show_node_labels=True,
                   show_edge_labels=False,
                   color_values=expansion_color_values)


if __name__ == '__main__':
    unittest.main()
