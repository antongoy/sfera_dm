import csv
import numpy as np
import random as rnd

from itertools import izip

from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score

__author__ = 'anton-goy'


class Classifier:
    def __init__(self, n_features, epsilon=0.001, eta0=0.001, alpha=0.01, num_iteration=1000):
        self.epsilon = np.full(n_features + 1, epsilon)
        self.eta0 = eta0
        self.n_iteration = num_iteration
        self.alpha = np.full(n_features + 1, alpha)
        self.alpha[0] = 0
        self.w = np.ones(n_features + 1)

    def __recompute_eta(self, cur_iter):
        return self.eta0 / (1 + self.eta0 * self.alpha * cur_iter)

    @staticmethod
    def __compute_gradient(w, x, y):
        return (sigmoid(np.dot(w, x)) - y) * x

    def fit(self, data_set, target_vars):
        n_data_samples = data_set.shape[0]
        norm_data_set = np.column_stack((np.ones(n_data_samples), data_set))

        rand_num = rnd.randint(0, n_data_samples - 1)

        k = 0

        cur_eta = self.__recompute_eta(cur_iter=0)
        cur_x = norm_data_set[rand_num]
        cur_y = target_vars[rand_num]
        cur_gradient = self.__compute_gradient(self.w, cur_x, cur_y)
        self.w = (1 - self.alpha) * self.w - cur_eta * cur_gradient
        k += 1

        while k < self.n_iteration:

            rand_num = rnd.randint(0, n_data_samples - 1)

            cur_eta = self.__recompute_eta(cur_iter=k)
            cur_x = norm_data_set[rand_num]
            cur_y = target_vars[rand_num]
            cur_gradient = self.__compute_gradient(self.w, cur_x, cur_y)
            self.w = (1 - self.alpha) * self.w - cur_eta * cur_gradient
            k += 1

    def predict_proba(self, test_data_set):
        if test_data_set.shape[1] == 1:
            norm_data_set = np.ones(test_data_set.shape[0] + 1)
            norm_data_set[1:] = test_data_set
            proba = sigmoid(np.dot(self.w, norm_data_set))
            return proba

        n_data_samples = test_data_set.shape[0]
        norm_data_set = np.column_stack((np.ones(n_data_samples), test_data_set))
        probabilities = []

        for obj in norm_data_set:
            proba = sigmoid(np.dot(self.w, obj))
            probabilities.append(1 - proba)

        return probabilities


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def transform_target_vars(target_vars, class_num):
    class_target_vars = np.copy(target_vars)
    class_target_vars[target_vars == class_num] = 0
    class_target_vars[target_vars != class_num] = 1

    return class_target_vars


def decision_function(first_predict, second_predict, third_predict):
    return np.array(map(lambda s: np.argmax(list(s)) + 1,
                        izip(first_predict, second_predict, third_predict)))


def main():

    with open('data_set.csv', 'rb') as data_file:
        data = csv.reader(data_file)

        data_set = map(lambda s: [float(i) for i in s], list(data))
        data_set = np.array(data_set, dtype=np.float64)

        data_set = shuffle(data_set)

        target_vars = np.array(map(lambda s: int(s), data_set[:, 0]))
        data_set = data_set[:, 1:]

        scale(data_set, copy=False)

        folds = 5
        cross_validation_kfold = KFold(data_set.shape[0], n_folds=folds)

        n_features = data_set.shape[1]

        first_target_vars = transform_target_vars(target_vars, class_num=1)
        second_target_vars = transform_target_vars(target_vars, class_num=2)
        third_target_vars = transform_target_vars(target_vars, class_num=3)

        first_classifier = Classifier(n_features, eta0=0.1)
        second_classifier = Classifier(n_features, eta0=0.1)
        third_classifier = Classifier(n_features, eta0=0.1)

        accuracy = 0
        recall = 0
        precision = 0

        for train_indices, test_indices in cross_validation_kfold:
            first_classifier.fit(data_set[train_indices], first_target_vars[train_indices])
            second_classifier.fit(data_set[train_indices], second_target_vars[train_indices])
            third_classifier.fit(data_set[train_indices], third_target_vars[train_indices])

            first_predict_vars = first_classifier.predict_proba(data_set[test_indices])
            second_predict_vars = second_classifier.predict_proba(data_set[test_indices])
            third_predict_vars = third_classifier.predict_proba(data_set[test_indices])

            predicted = decision_function(first_predict_vars,
                                          second_predict_vars,
                                          third_predict_vars)

            accuracy += accuracy_score(target_vars[test_indices], predicted)
            recall += recall_score(target_vars[test_indices], predicted)
            precision += precision_score(target_vars[test_indices], predicted)

        accuracy /= folds
        recall /= folds
        precision /= folds

        print 'Cross-validation: '
        print '\tAccuracy', accuracy
        print '\tRecall', recall
        print '\tPrecision', precision


if __name__ == '__main__':
    main()
