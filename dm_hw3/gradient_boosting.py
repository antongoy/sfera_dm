import csv
import copy
import numpy as np
import decision_tree as dtree
import matplotlib.pyplot as pl

from scipy.optimize import minimize_scalar
from itertools import izip
from random import sample, randint
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


__author__ = 'anton-goy'


def load_wine():
    with open('data_set.csv', 'rb') as data_file:
        data = csv.reader(data_file)

        data_set = map(lambda s: [float(i) for i in s], list(data))
        data_set = np.array(data_set, dtype=np.float64)

        target_vars = np.array(map(lambda s: int(s) - 1, data_set[:, 0]))
        data_set = data_set[:, 1:]

    return {'data': data_set, 'target': target_vars}


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def loss_function(x, y):
    return np.sum(-y * np.log(x) - (1 - y) * np.log(1 - x))


def gradient(x, y):
    return (x - y) / x * (1 - x)


def const_func(data_set, const):
    return np.full(data_set.shape[0], const)


def get_random_subspace(n_random_features, n_features):
    return sample(range(n_features), n_random_features)


class GradientBoostingClassifier():
    def __init__(self, n_trees=10, max_tree_depth=2, n_features=5):
        self.func = lambda _: 0
        self.trees = []
        self.tree_weights = []
        self.n_trees = n_trees
        self.max_tree_depth = max_tree_depth
        self.n_random_features = n_features

    def __ensemble_func(self, x):
        tree_results = np.array([b * tree.predict(x) for tree, b in izip(self.trees, self.tree_weights)])
        return self.func(x) + np.sum(tree_results, axis=0)

    def fit(self, train_set, targets):
        n_feature = train_set.shape[1]

        init = minimize_scalar(lambda x: loss_function(sigmoid(x), targets))
        self.func = lambda x: const_func(x, init.x)

        for i in range(self.n_trees):
            random_subspace = get_random_subspace(self.n_random_features, n_feature)
            temp = self.__ensemble_func(train_set)
            anti_gradients = - gradient(sigmoid(temp), targets)
            new_tree = dtree.DecisionTreeRegressor(max_depth=self.max_tree_depth)
            new_tree.fit(train_set[:, random_subspace], anti_gradients)
            weight = minimize_scalar(lambda b:
                                     loss_function(sigmoid(temp + b * new_tree.predict(train_set)), targets))
            self.trees.append(new_tree)
            self.tree_weights.append(weight.x)

    def predict(self, test_set):
        return 1 - sigmoid(self.__ensemble_func(test_set))


def transform_target_vars(target_vars, class_num):
    class_target_vars = np.copy(target_vars)
    class_target_vars[target_vars == class_num] = 0
    class_target_vars[target_vars != class_num] = 1

    return class_target_vars


def decision_function(first_predict, second_predict, third_predict):
    return np.array(map(lambda s: np.argmax(list(s)),
                        izip(first_predict, second_predict, third_predict)))


def main():
    all_targets = load_iris()['target']
    data_set = load_iris()['data']

    train_set, test_set, targets, targets_test = train_test_split(data_set, all_targets, train_size=0.9)

    targets_class = (transform_target_vars(targets, class_num=0),
                     transform_target_vars(targets, class_num=1),
                     transform_target_vars(targets, class_num=2))

    for n_trees in range(1, 150, 10):
        classifiers = (GradientBoostingClassifier(n_trees=n_trees, max_tree_depth=1, n_features=3),
                       GradientBoostingClassifier(n_trees=n_trees, max_tree_depth=1, n_features=3),
                       GradientBoostingClassifier(n_trees=n_trees, max_tree_depth=1, n_features=3))

        classifiers[0].fit(train_set, targets_class[0])
        classifiers[1].fit(train_set, targets_class[1])
        classifiers[2].fit(train_set, targets_class[2])

        predicts = (classifiers[0].predict(test_set),
                    classifiers[1].predict(test_set),
                    classifiers[2].predict(test_set))

        fin_predict = decision_function(predicts[0], predicts[1], predicts[2])

        print "Number of trees:", n_trees, ":", accuracy_score(targets_test, fin_predict)


if __name__ == '__main__':
    main()