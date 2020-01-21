# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:55:46 2019

@author: songl
"""

from __future__ import division

import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None

    # TODO: finish this.
    # raise NotImplemented()
    A1 = DecisionNode(None, None, lambda feature: feature[0]==1)
    A2 = DecisionNode(None, None, lambda feature: feature[1]==1)
    A3 = DecisionNode(None, None, lambda feature: feature[2]==1)
    A4 = DecisionNode(None, None, lambda feature: feature[3]==1)

    decision_tree_root = A1
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    decision_tree_root.right = A4
    A4.left = A2
    A4.right = A3
    A2.left = DecisionNode(None, None, None, 0)
    A2.right = DecisionNode(None, None, None, 1)
    A3.left = DecisionNode(None, None, None, 0)
    A3.right = DecisionNode(None, None, None, 1)

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    # raise NotImplemented()
    true_list = [x for x,y in zip(classifier_output, true_labels) if y==1]
    false_list = [x for x,y in zip(classifier_output, true_labels) if y==0]
    TP = Counter(true_list)[1]
    FN = Counter(true_list)[0]
    FP = Counter(false_list)[1]
    TN = Counter(false_list)[0]
    return [[TP,FN],[FP,TN]]


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    # raise NotImplemented()
    true_list = [x for x,y in zip(classifier_output, true_labels) if y==1]
    false_list = [x for x,y in zip(classifier_output, true_labels) if y==0]
    TP = Counter(true_list)[1]
    FP = Counter(false_list)[1]
    return TP / (TP + FP + 0.0)


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    # raise NotImplemented()
    true_list = [x for x,y in zip(classifier_output, true_labels) if y==1]
    TP = Counter(true_list)[1]
    FN = Counter(true_list)[0]
    return TP / (TP + FN + 0.0)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    # raise NotImplemented()
    true_list = [x for x,y in zip(classifier_output, true_labels) if y==1]
    false_list = [x for x,y in zip(classifier_output, true_labels) if y==0]
    TP = Counter(true_list)[1]
    TN = Counter(false_list)[0]
    return (TP + TN + 0.0) / (len(true_labels))


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    # raise NotImplemented()
    dic = Counter(class_vector)
    p0 = dic[0] / len(class_vector)
    p1 = dic[1] / len(class_vector)
    GI = 1-p0**2-p1**2
    return GI


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    # raise NotImplemented()
    previous = gini_impurity(previous_classes)
    current = 0
    total = sum([len(l) for l in current_classes])
    for lst in current_classes:
        num = len(lst)
        current += (num / total) * gini_impurity(lst)
    gain = previous - current
    return gain


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        # raise NotImplemented()

        # base cases
        class_dict = Counter(classes)
        if len(class_dict) == 1:
            return DecisionNode(None, None, None, classes[0])
        if depth == self.depth_limit and len(class_dict) == 2:
            majority = int(class_dict[1] >= class_dict[0])
            return DecisionNode(None, None, None, majority)

        # split continuous variables using mean value (equal-frequency method)
        mean_lst = np.mean(features, axis=0)
        new_features = []
        for lst in features:
            tmp = [int(i) for i in (lst >= mean_lst)]
            new_features.append(tmp)

        new_features = np.array(new_features)

        # calculate gini gain in the current level
        gain_lst = []
        for i in range(len(new_features[0])):
            this_feature = new_features[:,i]
            current_class = []
            for j in list(set(this_feature)):
                current_class.append([y for x,y in zip(this_feature, classes) if x == j])
            this_gain = gini_gain(classes, current_class)
            gain_lst.append(this_gain)
        gain_lst = np.divide(gain_lst, sum(gain_lst))
        # best feature given max gini gain
        best_feature_idx = np.argmax(gain_lst)
        best_feature = new_features[:,best_feature_idx]
        # 0 and 1 index list to separate classes
        zero_idx = [i for i in range(len(best_feature)) if best_feature[i]==0]
        one_idx = [i for i in range(len(best_feature)) if best_feature[i]==1]
        # increment of depth
        depth += 1
        # recursive steps
        left = self.__build_tree__(np.delete(features,zero_idx,0), [classes[k] for k in one_idx], depth)
        right = self.__build_tree__(np.delete(features,one_idx,0), [classes[k] for k in zero_idx], depth)

        return DecisionNode(left, right, lambda feature: feature[best_feature_idx] > mean_lst[best_feature_idx])



    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (list(list(int)): List of features.
        Return:
            A list of class labels.
        """

        # TODO: finish this.
        # raise NotImplemented()
        class_labels = [self.root.decide(feature) for feature in features]

        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
    """

    # TODO: finish this.
    # raise NotImplemented()

    k_folds = []
    total_features = dataset[0]
    # num_sample = len(total_features)
    # total_classes = np.asarray(dataset[1]).reshape(num_sample,1)
    # size = int(num_sample / k)
    # total_examples = np.concatenate((total_features, total_classes), axis=1)

    for i in range(k):

        previous_classes, this_classes = list(dataset[1]), []
        previous_examples, this_examples = np.copy(dataset[0]), []

        for j in range(int(len(dataset[-1]) / k)):
            rand = np.random.randint(0, len(previous_classes) - 1)
            this_examples.append(previous_examples[rand])
            previous_examples = np.delete(previous_examples, rand, 0)
            this_classes.append(previous_classes.pop(rand))
        test = (this_examples, this_classes)
        train = (previous_examples, previous_classes)
        this_fold = (train, test)
        k_folds.append(this_fold)

    return k_folds



class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.attr = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        # raise NotImplemented()

        # base cases
        class_dict = Counter(classes)
        if len(class_dict) == 1:
            return DecisionNode(None, None, None, classes[0])
        if depth == self.depth_limit and len(class_dict) == 2:
            majority = int(class_dict[1] >= class_dict[0])
            return DecisionNode(None, None, None, majority)

        # split continuous variables using mean value (equal-frequency method)
        mean_lst = np.mean(features, axis=0)
        new_features = []
        for lst in features:
            tmp = [int(i) for i in (lst >= mean_lst)]
            new_features.append(tmp)

        new_features = np.array(new_features)

        # calculate gini gain in the current level
        gain_lst = []
        for i in range(len(new_features[0])):
            this_feature = new_features[:,i]
            current_class = []
            for j in list(set(this_feature)):
                current_class.append([y for x,y in zip(this_feature, classes) if x == j])
            this_gain = gini_gain(classes, current_class)
            gain_lst.append(this_gain)
        gain_lst = np.divide(gain_lst, sum(gain_lst))
        # best feature given max gini gain
        best_feature_idx = np.argmax(gain_lst)
        best_feature = new_features[:,best_feature_idx]
        # 0 and 1 index list to separate classes
        zero_idx = [i for i in range(len(best_feature)) if best_feature[i]==0]
        one_idx = [i for i in range(len(best_feature)) if best_feature[i]==1]
        # increment of depth
        depth += 1
        # recursive steps
        left = self.__build_tree__(np.delete(features,zero_idx,0), [classes[k] for k in one_idx], depth)
        right = self.__build_tree__(np.delete(features,one_idx,0), [classes[k] for k in zero_idx], depth)

        return DecisionNode(left, right, lambda feature: feature[best_feature_idx] > mean_lst[best_feature_idx])

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        # raise NotImplemented()

        feature_len = len(features[:,0])
        attrib_num = len(features[0])
        sub_feature_num = int(feature_len * self.example_subsample_rate)
        sub_attrib_num = int(attrib_num * self.attr_subsample_rate)

        for i in range(0, self.num_trees):
            cur_features = []
            cur_classes = []

            for j in range(0, sub_feature_num):
                rand = np.random.randint(0, feature_len - 1)
                cur_features.append(features[rand])
                cur_classes.append(classes[rand])
            cur_features = np.array(cur_features)
            self.attr.append([])
            for j in range(0, sub_attrib_num):
                rand = np.random.randint(0, attrib_num - 1)
                if self.attr[i].count(rand) == 0:
                    self.attr[i].append(rand)
                # cur_features = np.delete(cur_features, rand, 1)
            # Use only the retrieved attribs
            cur_features = cur_features[:,self.attr[i]]
            self.trees.append(self.__build_tree__(cur_features, cur_classes))



    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (list(list(int)): List of features.
        """

        # TODO: finish this.
        # raise NotImplemented()
        class_labels = []
        for row in features:
            results = [0] * self.num_trees
            for i in range(0, self.num_trees):
                this_row = row[self.attr[i]]
                results[i] += self.trees[i].decide(this_row)
            class_labels.append(sum(results)/self.num_trees)
        class_labels = [int(round(x)) for x in class_labels]

        return class_labels


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        # raise NotImplemented()
        self.trees = []
        self.attr = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate


    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        # raise NotImplemented()

        # base cases
        class_dict = Counter(classes)
        if len(class_dict) == 1:
            return DecisionNode(None, None, None, classes[0])
        if depth == self.depth_limit and len(class_dict) == 2:
            majority = int(class_dict[1] >= class_dict[0])
            return DecisionNode(None, None, None, majority)

        # split continuous variables using mean value (equal-frequency method)
        mean_lst = np.mean(features, axis=0)
        new_features = []
        for lst in features:
            tmp = [int(i) for i in (lst >= mean_lst)]
            new_features.append(tmp)

        new_features = np.array(new_features)

        # calculate gini gain in the current level
        gain_lst = []
        for i in range(len(new_features[0])):
            this_feature = new_features[:,i]
            current_class = []
            for j in list(set(this_feature)):
                current_class.append([y for x,y in zip(this_feature, classes) if x == j])
            this_gain = gini_gain(classes, current_class)
            gain_lst.append(this_gain)
        gain_lst = np.divide(gain_lst, sum(gain_lst))
        # best feature given max gini gain
        best_feature_idx = np.argmax(gain_lst)
        best_feature = new_features[:,best_feature_idx]
        # 0 and 1 index list to separate classes
        zero_idx = [i for i in range(len(best_feature)) if best_feature[i]==0]
        one_idx = [i for i in range(len(best_feature)) if best_feature[i]==1]
        # increment of depth
        depth += 1
        # recursive steps
        left = self.__build_tree__(np.delete(features,zero_idx,0), [classes[k] for k in one_idx], depth)
        right = self.__build_tree__(np.delete(features,one_idx,0), [classes[k] for k in zero_idx], depth)

        return DecisionNode(left, right, lambda feature: feature[best_feature_idx] > mean_lst[best_feature_idx])


    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        # raise NotImplemented()
        feature_len = len(features[:,0])
        attrib_num = len(features[0])
        sub_feature_num = int(feature_len * self.example_subsample_rate)
        sub_attrib_num = int(attrib_num * self.attr_subsample_rate)

        for i in range(0, self.num_trees):
            cur_features = []
            cur_classes = []

            for j in range(0, sub_feature_num):
                rand = np.random.randint(0, feature_len - 1)
                cur_features.append(features[rand])
                cur_classes.append(classes[rand])
            cur_features = np.array(cur_features)
            self.attr.append([])
            for j in range(0, sub_attrib_num):
                rand = np.random.randint(0, attrib_num - 1)
                if self.attr[i].count(rand) == 0:
                    self.attr[i].append(rand)
                # cur_features = np.delete(cur_features, rand, 1)
            # Use only the retrieved attribs
            cur_features = cur_features[:,self.attr[i]]
            self.trees.append(self.__build_tree__(cur_features, cur_classes))


    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (list(list(int)): List of features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        # raise NotImplemented()
        class_labels = []
        for row in features:
            results = [0] * self.num_trees
            for i in range(0, self.num_trees):
                this_row = row[self.attr[i]]
                results[i] += self.trees[i].decide(this_row)
            class_labels.append(sum(results)/self.num_trees)
        class_labels = [int(round(x)) for x in class_labels]

        return class_labels


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Bonnie time to beat: 0.09 seconds.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        # raise NotImplemented()
        return data * data + data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Bonnie time to beat: 0.07 seconds
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        # raise NotImplemented()
        tmp = np.sum(data[:100], axis=1)
        idx = np.argmax(tmp)
        return (tmp[idx], idx)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Bonnie time to beat: 15 seconds
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        # raise NotImplemented()
        a = np.unique(data[data > 0], return_counts=True)
        return zip(a[0], a[1])
        
def return_your_name():
    # return your name
    # TODO: finish this
    return ("Bian Du")