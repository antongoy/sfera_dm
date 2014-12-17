import numpy as np

from itertools import tee, izip

__author__ = 'anton-goy'


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def compute_boundaries(feature_vector):
    return np.array([(u + v) / 2.0 for u, v in pairwise(np.unique(np.sort(feature_vector)))])


def split_dataset(train_set, feature, divide_value):
    left_set = []
    right_set = []

    for i, obj in enumerate(train_set):
        if obj[feature] < divide_value:
            left_set.append(i)
        else:
            right_set.append(i)

    return left_set, right_set


def regression_impurity(targets):
    return np.sum(np.power(targets - targets.mean(), 2))


class DecisionTreeRegressor():
    def __init__(self, max_depth=2):
        self.tree = {}
        self.max_depth = max_depth

    def fit(self, train_set, targets):
        self.__fit_inside(self.tree, 1, train_set, targets)

    def __fit_inside(self, tree, current_depth, train_set, targets):
        n_features = train_set.shape[1]
        n_samples = train_set.shape[0]

        dataset_impurity = regression_impurity(targets)

        best_impurity = None
        best_left_impurity = None
        best_right_impurity = None
        best_right_set = None
        best_left_set = None
        best_feature = None
        best_value = None

        for feature in range(n_features):
            boundaries = compute_boundaries(train_set[:, feature])
            for divide_value in boundaries:
                left_set, right_set = split_dataset(train_set, feature, divide_value)

                left_impurity = regression_impurity(targets[left_set])
                right_impurity = regression_impurity(targets[right_set])

                n_left = len(left_set)
                n_right = len(right_set)

                impurity = dataset_impurity - (float(n_left) / n_samples) * left_impurity - \
                           (float(n_right) / n_samples) * right_impurity

                if best_impurity < impurity:
                    best_impurity = impurity
                    best_left_impurity = left_impurity
                    best_right_impurity = right_impurity
                    best_left_set = left_set
                    best_right_set = right_set
                    best_feature = feature
                    best_value = divide_value

        tree[(best_feature, best_value)] = {'left': {}, 'right': {}}

        if best_left_impurity != 0 and current_depth != self.max_depth:
            self.__fit_inside(tree[(best_feature, best_value)]['left'],
                              current_depth + 1,
                              train_set[best_left_set],
                              targets[best_left_set])
        else:
            tree[(best_feature, best_value)]['left'] = targets[best_left_set].mean()

        if best_right_impurity != 0 and current_depth != self.max_depth:
            self.__fit_inside(tree[(best_feature, best_value)]['right'],
                              current_depth + 1,
                              train_set[best_right_set],
                              targets[best_right_set])
        else:
            tree[(best_feature, best_value)]['right'] = targets[best_right_set].mean()

    def predict(self, test_set):
        return np.array([self.__predict_inside(sample, self.tree) for sample in test_set])

    def __predict_inside(self, sample, tree):
        if type(tree) != type({}):
            return tree

        for node, subtree in tree.iteritems():
            feature = node[0]
            value = node[1]
            if sample[feature] < value:
                return self.__predict_inside(sample, subtree['left'])
            else:
                return self.__predict_inside(sample, subtree['right'])
