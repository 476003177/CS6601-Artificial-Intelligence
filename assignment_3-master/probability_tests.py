import unittest
from submission import *
"""
Contains various local tests for Assignment 3.
"""

class ProbabilityTests(unittest.TestCase):

    #Part 1a
    def test_network_setup(self):
        """Test that the power plant network has the proper number of nodes and edges."""
        power_plant = make_power_plant_net()
        nodes = power_plant.nodes()
        self.assertEqual(len(nodes), 5, msg="incorrect number of nodes")
        total_links = power_plant.number_of_edges()
        self.assertEqual(total_links, 5, msg="incorrect number of edges between nodes")

    #Part 1b
    def test_probability_setup(self):
        """Test that all nodes in the power plant network have proper probability distributions.
        Note that all nodes have to be named predictably for tests to run correctly."""
        # first test temperature distribution
        power_plant = set_probability(make_power_plant_net())
        T_node = power_plant.get_cpds('temperature')
        self.assertTrue(T_node is not None, msg='No temperature node initialized')

        T_dist = T_node.get_values()
        self.assertEqual(len(T_dist), 2, msg='Incorrect temperature distribution size')
        test_prob = T_dist[0]
        self.assertEqual(round(float(test_prob*100)), 80, msg='Incorrect temperature distribution')

        # then faulty gauge distribution
        F_G_node = power_plant.get_cpds('faulty gauge')
        self.assertTrue(F_G_node is not None, msg='No faulty gauge node initialized')

        F_G_dist = F_G_node.get_values()
        rows, cols = F_G_dist.shape
        self.assertEqual(rows, 2, msg='Incorrect faulty gauge distribution size')
        self.assertEqual(cols, 2, msg='Incorrect faulty gauge distribution size')
        test_prob1 = F_G_dist[0][1]
        test_prob2 = F_G_dist[1][0]
        self.assertEqual(round(float(test_prob2*100)), 5, msg='Incorrect faulty gauge distribution')
        self.assertEqual(round(float(test_prob1*100)), 20, msg='Incorrect faulty gauge distribution')

        # faulty alarm distribution
        F_A_node = power_plant.get_cpds('faulty alarm')
        self.assertTrue(F_A_node is not None, msg='No faulty alarm node initialized')
        F_A_dist = F_A_node.get_values()
        self.assertEqual(len(F_A_dist), 2, msg='Incorrect faulty alarm distribution size')

        test_prob = F_A_dist[0]

        self.assertEqual(round(float(test_prob*100)), 85, msg='Incorrect faulty alarm distribution')
        # gauge distribution
        # can't test exact probabilities because
        # order of probabilities is not guaranteed
        G_node = power_plant.get_cpds('gauge')
        self.assertTrue(G_node is not None, msg='No gauge node initialized')
        [cols, rows1, rows2] = G_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect gauge distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect gauge distribution size')
        self.assertEqual(cols,  2, msg='Incorrect gauge distribution size')

        # alarm distribution
        A_node = power_plant.get_cpds('alarm')
        self.assertTrue(A_node is not None, msg='No alarm node initialized')
        [cols, rows1, rows2] = A_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect alarm distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect alarm distribution size')
        self.assertEqual(cols,  2, msg='Incorrect alarm distribution size')

        try:
            power_plant.check_model()
        except:
            self.assertTrue(False, msg='Sum of the probabilities for each state is not equal to 1 or CPDs associated with nodes are not consistent with their parents')


    #Part 2a Test
    def test_games_network(self):
        """Test that the games network has the proper number of nodes and edges."""
        games_net = get_game_network()
        nodes = games_net.nodes()
        self.assertEqual(len(nodes), 6, msg='Incorrent number of nodes')
        total_links = games_net.number_of_edges()
        self.assertEqual(total_links, 6, 'Incorrect number of edges')

        # Now testing that all nodes in the games network have proper probability distributions.
        # Note that all nodes have to be named predictably for tests to run correctly.

        # First testing team distributions.
        # You can check this for all teams i.e. A,B,C (by replacing the first line for 'B','C')

        A_node = games_net.get_cpds('A')
        self.assertTrue(A_node is not None, 'Team A node not initialized')
        A_dist = A_node.get_values()
        self.assertEqual(len(A_dist), 4, msg='Incorrect distribution size for Team A')
        test_prob = A_dist[0]
        test_prob2 = A_dist[2]
        self.assertEqual(round(float(test_prob*100)),  15, msg='Incorrect distribution for Team A')
        self.assertEqual(round(float(test_prob2*100)), 30, msg='Incorrect distribution for Team A')

        # Now testing match distributions.
        # You can check this for all matches i.e. AvB,BvC,CvA (by replacing the first line)
        AvB_node = games_net.get_cpds('AvB')
        self.assertTrue(AvB_node is not None, 'AvB node not initialized')

        AvB_dist = AvB_node.get_values()
        [cols, rows1, rows2] = AvB_node.cardinality
        self.assertEqual(rows1, 4, msg='Incorrect match distribution size')
        self.assertEqual(rows2, 4, msg='Incorrect match distribution size')
        self.assertEqual(cols,  3, msg='Incorrect match distribution size')

        flag1 = True
        flag2 = True
        flag3 = True
        for i in range(0, 4):
            for j in range(0,4):
                x = AvB_dist[:,(i*4)+j]
                if i==j:
                    if x[0]!=x[1]:
                        flag1=False
                if j>i:
                    if not(x[1]>x[0] and x[1]>x[2]):
                        flag2=False
                if j<i:
                    if not (x[0]>x[1] and x[0]>x[2]):
                        flag3=False

        self.assertTrue(flag1, msg='Incorrect match distribution for equal skill levels')
        self.assertTrue(flag2 and flag3, msg='Incorrect match distribution: teams with higher skill levels should have higher win probabilities')

    #Part 2b Test
    def test_posterior(self):
        posterior = calculate_posterior(get_game_network())

        self.assertTrue(abs(posterior[0]-0.25)<0.01 and abs(posterior[1]-0.42)<0.01 and abs(posterior[2]-0.31)<0.01, msg='Incorrect posterior calculated')

if __name__ == '__main__':
    unittest.main()
