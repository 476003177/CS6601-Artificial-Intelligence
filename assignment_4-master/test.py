# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 23:23:57 2019

@author: songl
"""

import unittest
import submission as dt
import numpy as np
import time
class DecisionTreePart2Tests(unittest.TestCase):
    """Tests for Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """

        self.restaurant = {'restaurants': [0] * 6 + [1] * 6,
                           'split_patrons': [[0, 0],
                                             [1, 1, 1, 1],
                                             [1, 1, 0, 0, 0, 0]],
                           'split_food_type': [[0, 1],
                                               [0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 1, 1]]}

        self.dataset = dt.load_csv('part23_data.csv')
        self.train_features, self.train_classes = self.dataset

    def test_gini_impurity_max(self):
        """Test maximum gini impurity.

        Asserts:
            gini impurity is 0.5.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 0, 0, 0])

        assert  .500 == round(gini_impurity, 3)

    def test_gini_impurity_min(self):
        """Test minimum gini impurity.

        Asserts:
            entropy is 0.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 1, 1, 1])

        assert 0 == round(gini_impurity, 3)

    def test_gini_impurity(self):
        """Test gini impurity.

        Asserts:
            gini impurity is matched as expected.
        """

        gini_impurity = dt.gini_impurity([1, 1, 0, 0, 0, 0])

        assert round(4. / 9., 3) == round(gini_impurity, 3)
    
    def test_gini_gain_max(self):
        """Test maximum gini gain.

        Asserts:
            gini gain is 0.5.
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 1], [0, 0, 0]])
        

        assert .500 == round(gini_gain, 3)

    def test_gini_gain(self):
        """Test gini gain.

        Asserts:
            gini gain is within acceptable bounds
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 0], [1, 0, 0]])
        
        assert 0.056 == round(gini_gain, 3)

    def test_gini_gain_restaurant_patrons(self):
        """Test gini gain using restaurant patrons.

        Asserts:
            gini gain rounded to 3 decimal places matches as expected.
        """

        gain_patrons = dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_patrons'])
        #print(round(gain_patrons, 3))
        assert round(gain_patrons, 3) == 0.278

    def test_gini_gain_restaurant_type(self):
        """Test gini gain using restaurant food type.

        Asserts:
            gini gain is 0.
        """

        gain_type = round(dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_food_type']), 2)
        #print(round(gain_type, 3))
        assert gain_type == 0.00
        
    def test_decision_tree_all_data(self):
        """Test decision tree classifies all data correctly.

        Asserts:
            classification is 100% correct.
        """

        tree = dt.DecisionTree()
        tree.fit(self.train_features, self.train_classes)
        output = tree.classify(self.train_features)

        assert (output == self.train_classes).all()
        
    def test_k_folds_test_set_count(self):
        """Test k folds returns the correct test set size.

        Asserts:
            test set size matches as expected.
        """

        example_count = len(self.train_features)
        k = 10
        test_set_count = example_count // k
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold

            assert len(test_set[0]) == test_set_count

    def test_k_folds_training_set_count(self):
        """Test k folds returns the correct training set size.

        Asserts:
            training set size matches as expected.
        """

        example_count = len(self.train_features)
        k = 10
        training_set_count = example_count - (example_count // k)
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold

            assert len(training_set[0]) == training_set_count
        
if __name__ == '__main__':
    unittest.main()