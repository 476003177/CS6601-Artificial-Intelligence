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

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """
    
    # TODO: finish this.
    decision_tree_root=DecisionNode(None,None,lambda feature:feature[0]==1)
    decision_tree_root.left=DecisionNode(None,None,None,1)
    decision_tree_root.right=DecisionNode(None,None,lambda feature:feature[2]+feature[3]==1)
    decision_tree_root.right.left=DecisionNode(None,None,None,0)
    decision_tree_root.right.right=DecisionNode(None,None,None,1)
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
    tp=0
    fn=0
    fp=0
    tn=0
    for i in range(len(true_labels)):
        if true_labels[i]==1 and classifier_output[i]==1:
            tp+=1
        elif true_labels[i]==1 and classifier_output[i]==0:
            fn+=1
        elif true_labels[i]==0 and classifier_output[i]==1:
            fp+=1
        else:
            tn+=1
    return [[tp,fn],[fp,tn]]


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
    tp=0
    fn=0
    fp=0
    tn=0
    for i in range(len(true_labels)):
        if true_labels[i]==1 and classifier_output[i]==1:
            tp+=1
        elif true_labels[i]==1 and classifier_output[i]==0:
            fn+=1
        elif true_labels[i]==0 and classifier_output[i]==1:
            fp+=1
        else:
            tn+=1
    return tp/(tp+fp)


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
    tp=0
    fn=0
    fp=0
    tn=0
    for i in range(len(true_labels)):
        if true_labels[i]==1 and classifier_output[i]==1:
            tp+=1
        elif true_labels[i]==1 and classifier_output[i]==0:
            fn+=1
        elif true_labels[i]==0 and classifier_output[i]==1:
            fp+=1
        else:
            tn+=1
    return tp/(tp+fn)


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
    tp=0
    fn=0
    fp=0
    tn=0
    for i in range(len(true_labels)):
        if true_labels[i]==1 and classifier_output[i]==1:
            tp+=1
        elif true_labels[i]==1 and classifier_output[i]==0:
            fn+=1
        elif true_labels[i]==0 and classifier_output[i]==1:
            fp+=1
        else:
            tn+=1
    return (tp+tn)/(len(true_labels))


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
    c=Counter(class_vector)
    num=c[0]
    p=num/len(class_vector)
    return 2*p*(1-p)


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    # you need to consider the prob of each current class
    previous_impurity=gini_impurity(previous_classes)
    current_impurity=0
    for current_class in current_classes:
        current_impurity+=gini_impurity(current_class)*len(current_class)/len(previous_classes)
    return previous_impurity-current_impurity


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
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        
        # TODO: finish this.
        m=np.size(features,axis=0)
        n=np.size(features,axis=1)
        c=Counter(classes)
        if len(c)==1:
            return DecisionNode(None,None,None,classes[0])
        if depth==self.depth_limit:
            if (c[0]>c[1]):
                return DecisionNode(None,None,None,0)
            else:
                return DecisionNode(None,None,None,1)
        # try to split the features using median
        # 0 is vertical axis, 1 is horizontal axis 
        
        mean=np.mean(features,axis=0)
        left=[]
        right=[]
        max_gain=0
        max_gain_index=0
        for i in range(n):
            left_class=classes[features[:,i]<mean[i]]
            right_class=classes[features[:,i]>=mean[i]]
            if np.size(left_class)!=0 and np.size(right_class)!=0:
                gain=gini_gain(classes,[left_class.tolist(),right_class.tolist()])
            else:
                gain=0 
            if max_gain<gain:
                max_gain=gain
                max_gain_index=i
        split_feature=features[:,max_gain_index]
        # you should never compare your feature with the max gini gain
        left_features=features[split_feature<mean[max_gain_index]]
        right_features=features[split_feature>=mean[max_gain_index]]
        left_classes=classes[split_feature<mean[max_gain_index]]
        right_classes=classes[split_feature>=mean[max_gain_index]]
        if np.size(left_classes)==0 or np.size(right_classes)==0:
            if (c[0]>c[1]):
                return DecisionNode(None,None,None,0)
            else:
                return DecisionNode(None,None,None,1)
        left=self.__build_tree__(left_features,left_classes,depth+1)
        right=self.__build_tree__(right_features,right_classes,depth+1)
        return DecisionNode(left,right,lambda features:features[max_gain_index]<mean[max_gain_index])
        """
        maxf=np.amax(features,axis=0)
        minf=np.amin(features,axis=0)
        max_gain=0
        max_gain_index=0
        max_gain_attr=0
        max_gain_attr_k=0
        max_gain_k=0
        for i in range(n):
            diff=(maxf[i]-minf[i])/10
            for k in range(1,10):
                left_class=classes[features[:,i]<(minf[i]+diff*k)]
                right_class=classes[features[:,i]>=(minf[i]+diff*k)]
                if np.size(left_class)!=0 and np.size(right_class)!=0:
                    gain=gini_gain(classes,[left_class.tolist(),right_class.tolist()])
                else:
                    gain=0  
                if max_gain_attr<gain:
                    max_gain_attr=gain
                    max_gain_attr_k=k
            if max_gain<max_gain_attr:
                max_gain=max_gain_attr
                max_gain_index=i
                max_gain_k=max_gain_attr_k
        split_feature=features[:,max_gain_index]
        diff=(maxf[max_gain_index]-minf[max_gain_index])/10
        # you should never compare your feature with the max gini gain
        left_features=features[split_feature<(minf[max_gain_index]+diff*max_gain_k)]
        right_features=features[split_feature>=(minf[max_gain_index]+diff*max_gain_k)]
        left_classes=classes[split_feature<(minf[max_gain_index]+diff*max_gain_k)]
        right_classes=classes[split_feature>=(minf[max_gain_index]+diff*max_gain_k)]
        left=self.__build_tree__(left_features,left_classes,depth+1)
        right=self.__build_tree__(right_features,right_classes,depth+1)
        return DecisionNode(left,right,lambda features:features[max_gain_index]<(minf[max_gain_index]+diff*max_gain_k))
        """
        
        
        
    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = []
        m=np.size(features,axis=0)
        # TODO: finish this.
        for i in range(m):
            class_labels.append(self.root.decide(features[i,:]))
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    features=dataset[0]
    num_examples=np.size(features,axis=0)
    test_size=int(np.floor(num_examples/k))
    sample_index=np.random.choice(num_examples,k*test_size,replace=False)
    k_folds=[]
    train_classes=dataset[1]
    train_features=dataset[0]
    for i in range(k):
        # you need to keep total part the same to maintain the meaning of randomly generated index
        # if you delete before you add next to fold, the index is changed
        # you need to delete all together once
        rand=sample_index[i*test_size:(i+1)*test_size]
        this_feature=np.take(train_features,rand,axis=0)
        this_class=np.take(train_classes,rand)
        train_features = np.delete(train_features, rand, 0)
        train_classes=np.delete(train_classes, rand, 0)
        test=(this_feature,this_class)
        train=(train_features,train_classes)
        k_folds.append((train,test))
        train_classes=dataset[1]
        train_features=dataset[0]
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
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.attr_index=[]

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        m=np.size(features,axis=0)
        n=np.size(features,axis=1)
        for i in range(self.num_trees):
            sub_example_index=np.random.choice(m,int(m*self.example_subsample_rate),replace=True)
            sub_example=np.take(features,sub_example_index,0)
            sub_example_class=np.take(classes,sub_example_index)
            sub_attr_index=np.random.choice(n,int(n*self.attr_subsample_rate),replace=False)
            sub_attr_example=np.take(sub_example,sub_attr_index,1)
            tree=DecisionTree(self.depth_limit)
            tree.fit(sub_attr_example,sub_example_class)
            self.trees.append(tree)
            self.attr_index.append(sub_attr_index)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        class_labels = []
        m=np.size(features,axis=0)
        # TODO: finish this.
        for i in range(m):
            vote=[]
            for j in range(len(self.trees)):
                index=self.attr_index[j]
                feature=features[i,:]
                sub_feature=np.take(feature,index)
                vote.append(self.trees[j].root.decide(sub_feature))
                #vote.append(self.trees[j].root.decide(features[i,:]))
                # pass corresponding sub attributes to the particular tree 
                # we train use a and we classify only using a
            c=Counter(vote)
            class_labels.append(c.most_common(1)[0][0])
        return class_labels


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        # the depth of tree can't exceed the sub
        self.forest=RandomForest(20,10,0.8,0.8)

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        self.forest.fit(features,classes)

    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        return self.forest.classify(features)


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
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        return data*data+data

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
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this. 
        # 1 is horizontal axis
        data=data[0:100,:]
        sums=np.sum(data,axis=1)
        max_sum=np.amax(sums)
        max_sum_index=np.argmax(sums)
        return max_sum, max_sum_index
        

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
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        a=data.flatten()
        b=a[a>0]
        unique, counts = np.unique(b, return_counts=True)
        return np.asarray((unique, counts)).T.tolist()

def return_your_name():
    # return your name
    # TODO: finish this
    return "Xufan Song"
