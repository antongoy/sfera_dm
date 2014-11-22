import argparse
import csv
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from itertools import izip

from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.metrics import accuracy_score, recall_score, precision_score

__author__ = 'anton-goy'


class Classifier:
    def __init__(self, data_set, target_vars, class_value, basis_func='identity', epsilon=0.001, eta=0.0):
        self.class_value = class_value
        self.basis_func = basis_func
        self.SGD = SGDClassifier(loss='log', penalty='l2', fit_intercept=False,
                                 epsilon=epsilon, eta0=eta, learning_rate='optimal')

        self.target_vars = np.copy(target_vars)
        for i in xrange(len(self.target_vars)):
            if self.target_vars[i] != self.class_value:
                self.target_vars[i] = 2
            else:
                self.target_vars[i] = 1

        if self.basis_func == 'quad':
            self.data_set = np.power(data_set, 2)
        elif self.basis_func == 'identity':
            self.data_set = data_set
        elif self.basis_func == 'rbf':
            self.data_set = rbf(data_set, 0, 1)

    def train(self, train_indices):
        self.SGD.fit(self.data_set[train_indices], self.target_vars[train_indices])

    def predict_proba(self, test_indices):
        return self.SGD.predict_proba(self.data_set[test_indices])

    def predict(self, test_indices):
        return self.SGD.predict(self.data_set[test_indices])

    def score(self, test_indices):
        return self.SGD.score(self.data_set[test_indices], self.target_vars[test_indices])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rbf(x, mean, var):
    return np.exp(-np.power(x - mean, 2) / 2 * np.power(var, 2))


def parse():
    parser = argparse.ArgumentParser(description="Logistic Regression with L2 Regularization based on SGD")
    parser.add_argument('-f', '--func', dest='basis_function', action='store',
                        choices=['identity', 'quad', 'rbf'], help='Basis function', default='identity')
    return parser.parse_args()


def decision_function(first_predict, second_predict, third_predict):
    return np.array(map(lambda s: np.argmax(list(s)) + 1,
                        izip(first_predict[:, 0], second_predict[:, 0], third_predict[:, 0])))



def main():
    args = parse()

    with open('data_set.csv', 'rb') as data_file:
        data = csv.reader(data_file)
        data_set = map(lambda s: [float(i) for i in s], list(data))
        data_set = np.array(data_set)
        data_set = shuffle(data_set, random_state=0)

        target_vars = np.array(map(lambda s: int(s), data_set[:, 0]))
        data_set = data_set[:, 1:]

        data_set, validate_set, target_vars, validate_target_vars = \
            train_test_split(data_set, target_vars, train_size=0.85)

        scale(data_set, copy=False)
        scale(validate_set, copy=False)

        folds = 5
        cross_validation_kf = KFold(data_set.shape[0], n_folds=folds)

        first_class_classifier = Classifier(data_set=data_set, target_vars=target_vars,
                                            class_value=1, basis_func=args.basis_function)
        second_class_classifier = Classifier(data_set=data_set, target_vars=target_vars,
                                             class_value=2, basis_func=args.basis_function)
        third_class_classifier = Classifier(data_set=data_set, target_vars=target_vars,
                                            class_value=3, basis_func=args.basis_function)

        accuracy = 0
        recall = 0
        precision = 0

        for train_indices, test_indices in cross_validation_kf:
            first_class_classifier.train(train_indices)
            second_class_classifier.train(train_indices)
            third_class_classifier.train(train_indices)

            first_class_predict_vars = first_class_classifier.predict_proba(test_indices)
            second_class_predict_vars = second_class_classifier.predict_proba(test_indices)
            third_class_predict_vars = third_class_classifier.predict_proba(test_indices)

            predicted = decision_function(first_class_predict_vars,
                                          second_class_predict_vars,
                                          third_class_predict_vars)
            
            accuracy += accuracy_score(target_vars[test_indices], predicted)
            recall += recall_score(target_vars[test_indices], predicted)
            precision += precision_score(target_vars[test_indices], predicted)

            #third_accuracy += third_class_classifier.score(test_indices)
            #second_accuracy += second_class_classifier.score(test_indices)
            #third_accuracy += third_class_classifier.score(test_indices)

        accuracy /= folds
        recall /= folds
        precision /= folds

        print 'Cross-validation: '
        print '\tAccuracy', accuracy
        print '\tRecall', recall
        print '\tPrecision', precision

        indices = range(len(validate_set))
        first_class_classifier.data_set = validate_set
        first_class_classifier.target_vars = validate_target_vars
        
        second_class_classifier.data_set = validate_set
        second_class_classifier.target_vars = validate_target_vars
        
        third_class_classifier.data_set = validate_set
        third_class_classifier.target_vars = validate_target_vars
        
        validate_predict = decision_function(first_class_classifier.predict_proba(indices), 
                                             second_class_classifier.predict_proba(indices),
                                             third_class_classifier.predict_proba(indices))

        print 'Accuracy after validation:', accuracy_score(validate_target_vars, validate_predict)
        fig = plt.figure()
        for i in range(4):
            fig.add_subplot(2,2,i)

            col1 = np.random.randint(0, validate_set.shape[1])
            col2 = np.random.randint(0, validate_set.shape[1])
            valid_set = validate_set[:, [col1, col2]]

            plot_data_set = np.column_stack((validate_predict, valid_set))

            plot_data_set = plot_data_set[plot_data_set[:, 0].argsort()]

            first_plot = np.array([row for row in plot_data_set if row[0] == 1])
            second_plot = np.array([row for row in plot_data_set if row[0] == 2])
            third_plot = np.array([row for row in plot_data_set if row[0] == 3])



            plt.scatter(first_plot[:, 1:2], first_plot[:, 2:],  marker='o')
            plt.scatter(second_plot[:, 1:2], second_plot[:, 2:],  marker='v', c='r')
            plt.scatter(third_plot[:, 1:2], third_plot[:, 2:],  marker='x', c='g')

        plt.show()


if __name__ == '__main__':
    main()
