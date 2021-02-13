import numpy as np
from sklearn.base import BaseEstimator
import collections
from scipy import stats




def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    if y.shape[0] != 0:
        p = np.sum(y, axis=0)/y.shape[0]

        E = - np.sum(p*np.log(p + EPS))
    else:
        E = 1
    return E
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    if y.shape[0] != 0:
        p = np.sum(y, axis=0)/y.shape[0]

        G = 1 - np.sum(p**2)
        
    else: 
        G = 1
    
    return G
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    if y.shape[0] != 0:
        V = np.sum((y - np.mean(y))**2)/y.shape[0]
    else:
        V = 10**10
    
    return V

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    if y.shape[0] != 0:
        M = np.sum(np.abs(y - np.median(y)))/y.shape[0]
    else:
        M = 10**10
    
    return M


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        self.end = 0
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        X_left = X_subset[X_subset[:,feature_index]<threshold,:]
        X_right = X_subset[X_subset[:,feature_index]>=threshold,:]
        
        y_left = y_subset[X_subset[:,feature_index]<threshold,:]
        y_right = y_subset[X_subset[:,feature_index]>=threshold,:]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        y_left = y_subset[X_subset[:,feature_index]<threshold,:]
        y_right = y_subset[X_subset[:,feature_index]>=threshold,:]
        
#         if self.criterion_name in ['mad_median','variance']:
#             y_left = one_hot_encode(self.n_classes,y_left)
#             y_right = one_hot_encode(self.n_classes,y_right)
        
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        
        
        score = {}
        for feature_index in range(X_subset.shape[1]):
            score_thr = {}
            for threshold in np.unique(X_subset[:,feature_index]):
                y_left, y_right = self.make_split_only_y(feature_index,
                                                         threshold, 
                                                         X_subset, y_subset)
                if (y_left.shape[0] == 0) or (y_right.shape[0] == 0):
                    continue
                else:
                    if self.criterion_name == 'mad_median':
                        H_l = mad_median(y_left)
                        H_r = mad_median(y_right)
#                         H_q = mad_median(y_subset)
                        

                    if self.criterion_name == 'variance':
                        H_l = variance(y_left)
                        H_r = variance(y_right)
#                         H_q = variance(y_subset)
                        

                    if self.criterion_name == 'gini':
                        H_l = gini(y_left)
                        H_r = gini(y_right)
#                         H_q = gini(y_subset)
                        

                    if self.criterion_name == 'entropy':
                        H_l = entropy(y_left)
                        H_r = entropy(y_right)
#                         H_q = entropy(y_subset)
                        

                    criteria = y_left.shape[0]/y_subset.shape[0]*H_l + \
                               y_right.shape[0]/y_subset.shape[0]*H_r
                    
#                     criteria = H_q - y_left.shape[0]/y_subset.shape[0]*H_l - \
#                                y_right.shape[0]/y_subset.shape[0]*H_r
                
                    score[(feature_index,threshold)] = criteria
        feature_index, threshold = min(score, key=score.get)
        
        return feature_index, threshold
    
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        
        if self.criterion_name in ['gini','entropy']:      
    #         print("Классы",len(np.unique(one_hot_decode(y_subset))))
    #         print('Размер выборки',X_subset.shape[0])
#             print('Глубина', self.depth)
            if (X_subset.shape[0] >= self.min_samples_split) & \
            (len(np.unique(one_hot_decode(y_subset))) > 1) & \
            (self.depth <= self.max_depth):
                self.depth += 1
                bestsplit = self.choose_best_split(X_subset,y_subset)
                node = Node(bestsplit[0],bestsplit[1])
                (X_left, y_left), (X_right, y_right) = \
                self.make_split(bestsplit[0],bestsplit[1],X_subset,y_subset)
                node.left_child = self.make_tree(X_left, y_left)
                node.right_child = self.make_tree(X_right, y_right)

                return node

            else:
    #             print("!!!!!!!!!!")
                self.depth -= 1
    #             print('!!Глубина', self.depth)

                return y_subset
        if self.criterion_name in ['mad_median','variance']:
            if (X_subset.shape[0] >= self.min_samples_split) & \
            (self.depth <= self.max_depth):
                self.depth += 1
                bestsplit = self.choose_best_split(X_subset,y_subset)
                node = Node(bestsplit[0],bestsplit[1])
                (X_left, y_left), (X_right, y_right) = \
                self.make_split(bestsplit[0],bestsplit[1],X_subset,y_subset)
                node.left_child = self.make_tree(X_left, y_left)
                node.right_child = self.make_tree(X_right, y_right)

                return node

            else:
    #             print("!!!!!!!!!!")
                self.depth -= 1
    #             print('!!Глубина', self.depth)

                return y_subset
        
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification =\
        self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
        self.depth = 0
        self.root = self.make_tree(X, y)
    
    
    def split(self, tree, row):
        if type(tree) == np.ndarray:
            return tree
        
        if row[tree.feature_index] < tree.value:
            return self.split(tree.left_child, row)
        else:
            return self.split(tree.right_child, row)
        
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        tree = self.root
        
        if self.criterion_name in ['gini','entropy']:
            y_predicted = np.empty((0,1), int)
            for row in X:
                list_predict = self.split(tree, row)
                y_predicted = np.concatenate((y_predicted,
                    stats.mode(one_hot_decode(list_predict))[0]
                                             ), axis=0)
            
            
        if self.criterion_name in ['mad_median','variance']:
            y_predicted = np.empty((0,1), int)
            for row in X:
                list_predict = self.split(tree, row)
                y_predicted = np.concatenate((y_predicted,
                                    np.array([[np.mean(list_predict)]])
                                             ), axis=0)
            
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        tree = self.root
        y_predicted_probs = np.empty((0,10), float)

        for row in X:
            list_predict = self.split(tree, row)
            y_predicted_probs = np.concatenate((y_predicted_probs,
                [np.sum(list_predict, axis=0)/list_predict.shape[0]]
                                         ), axis=0)
            
        return y_predicted_probs
