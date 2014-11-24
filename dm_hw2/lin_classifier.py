import csv
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

from itertools import izip
from scipy.integrate import trapz
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold

__author__ = 'anton-goy'


class Classifier:
    def __init__(self, n_features, epsilon=0.00001, eta0=0.001, alpha=1, num_iteration=1000):
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
        self.w = (1 - cur_eta * self.alpha) * self.w - cur_eta * cur_gradient
        k += 1

        while k < self.n_iteration:
            rand_num = rnd.randint(0, n_data_samples - 1)

            cur_eta = self.__recompute_eta(cur_iter=k)
            cur_x = norm_data_set[rand_num]
            cur_y = target_vars[rand_num]
            cur_gradient = self.__compute_gradient(self.w, cur_x, cur_y)
            self.w = (1 - cur_eta * self.alpha) * self.w - cur_eta * cur_gradient
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

    def predict_threshold(self, test_data_set, threshold):
        probabilities = self.predict_proba(test_data_set)

        predict_thres = []

        for proba in probabilities:
            if proba > threshold:
                predict_thres.append(0)
            else:
                predict_thres.append(1)

        return predict_thres


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def metrics(target_vars, predict_vars, label):
    tp = tn = fp = fn = 0

    for target, predict in izip(target_vars, predict_vars):
        if target == predict and target == label:
            tp += 1
        if target == predict and target != label:
            tn += 1
        if target != predict and target != label:
            fp += 1
        if target != predict and target == label:
            fn += 1

    return tp, tn, fp, fn


def accuracy_compute(target_vars, predict_vars):
    tp1, tn1, fp1, fn1 = metrics(target_vars, predict_vars, label=1)
    tp2, tn2, fp2, fn2 = metrics(target_vars, predict_vars, label=2)
    tp3, tn3, fp3, fn3 = metrics(target_vars, predict_vars, label=3)

    accuracy1 = float(tp1 + tn1) / (tp1 + tn1 + fp1 + fn1)
    accuracy2 = float(tp2 + tn2) / (tp2 + tn2 + fp2 + fn2)
    accuracy3 = float(tp3 + tn3) / (tp3 + tn3 + fp3 + fn3)

    return (accuracy1 + accuracy2 + accuracy3) / 3.0


def recall_compute(target_vars, predict_vars, label):
    tp, tn, fp, fn = metrics(target_vars, predict_vars, label=label)

    return float(tp) / (tp + fn)


def precision_compute(target_vars, predict_vars, label):
    tp, tn, fp, fn = metrics(target_vars, predict_vars, label=label)

    return float(tp) / (tp + fp)


def tpr_compute(target_vars, predict_vars, label):
    tp, _, _, fn = metrics(target_vars, predict_vars, label=label)

    return float(tp) / (tp + fn)


def fpr_compute(target_vars, predict_vars, label):
    _, tn, fp, _ = metrics(target_vars, predict_vars, label=label)

    return float(fp) / (fp + tn)


def transform_target_vars(target_vars, class_num):
    class_target_vars = np.copy(target_vars)
    class_target_vars[target_vars == class_num] = 0
    class_target_vars[target_vars != class_num] = 1

    return class_target_vars


def decision_function(first_predict, second_predict, third_predict):
    return np.array(map(lambda s: np.argmax(list(s)) + 1,
                        izip(first_predict, second_predict, third_predict)))


def area_compute(fprs, tprs):
    S = 0
    for i in xrange(len(fprs)):
        if i == 0:
            continue
        if i == 1:
            S += 0.5 * fprs[i] * tprs[i]
            continue
        S += 0.5 * (tprs[i-1] - tprs[i-2] + tprs[i]) * (fprs[i] - fprs[i-1])

    return S


def main():
    with open('data_set.csv', 'rb') as data_file:
        data = csv.reader(data_file)

        data_set = map(lambda s: [float(i) for i in s], list(data))
        data_set = np.array(data_set, dtype=np.float64)

        data_set = shuffle(data_set)

        target_vars = np.array(map(lambda s: int(s), data_set[:, 0]))
        data_set = data_set[:, 1:]

        scale(data_set, copy=False)
        #data_set,  valid_set, target_vars, valid_target = train_test_split(data_set, target_vars,  test_size=0.9)
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
        first_recall = 0
        second_recall = 0
        third_recall = 0
        first_precision = 0
        second_precision = 0
        third_precision = 0

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

            accuracy += accuracy_compute(target_vars[test_indices], predicted)
            first_recall += recall_compute(target_vars[test_indices], predicted, label=1)
            second_recall += recall_compute(target_vars[test_indices], predicted, label=2)
            third_recall += recall_compute(target_vars[test_indices], predicted, label=3)
            first_precision += precision_compute(target_vars[test_indices], predicted, label=1)
            second_precision += precision_compute(target_vars[test_indices], predicted, label=2)
            third_precision += precision_compute(target_vars[test_indices], predicted, label=3)

        accuracy /= folds
        first_recall /= folds
        second_recall /= folds
        third_recall /= folds
        first_precision /= folds
        second_precision /= folds
        third_precision /= folds

        thresholds = np.linspace(0, 1, num=20)
        first_tprs = []
        first_fprs = []
        
        second_tprs = []
        second_fprs = []
        
        third_tprs = []
        third_fprs = []

        for thres in thresholds:
            predict_threshold = first_classifier.predict_threshold(data_set, thres)
            first_tpr = tpr_compute(first_target_vars, predict_threshold, label=1)
            first_fpr = fpr_compute(first_target_vars, predict_threshold, label=1)
            first_tprs.append(first_tpr)
            first_fprs.append(first_fpr)
            
        for thres in thresholds:
            predict_threshold = second_classifier.predict_threshold(data_set, thres)
            second_tpr = tpr_compute(second_target_vars, predict_threshold, label=1)
            second_fpr = fpr_compute(second_target_vars, predict_threshold, label=1)
            second_tprs.append(second_tpr)
            second_fprs.append(second_fpr)
            
        for thres in thresholds:
            predict_threshold = third_classifier.predict_threshold(data_set, thres)
            third_tpr = tpr_compute(third_target_vars, predict_threshold, label=1)
            third_fpr = fpr_compute(third_target_vars, predict_threshold, label=1)
            third_tprs.append(third_tpr)
            third_fprs.append(third_fpr)

        plt.figure(figsize=(20, 10))

        plt.subplot(131)
        plt.plot(first_fprs, first_tprs)

        plt.subplot(132)
        plt.plot(second_fprs, second_tprs)

        plt.subplot(133)
        plt.plot(third_fprs, third_tprs)

        plt.show()

        print '\tAccuracy', accuracy
        print '\tRecall For 1st class', first_recall
        print '\tRecall For 2nd class', second_recall
        print '\tRecall For 3d class', third_recall
        print '\tPrecision For 1st class', first_precision
        print '\tPrecision For 2nd class', second_precision
        print '\tPrecision For 3d class', third_precision
        print '\n'
        print '\tFirst AUC', trapz(first_tprs, first_fprs)
        print '\tSecond AUC', trapz(second_tprs, second_fprs)
        print '\tThird AUC', trapz(third_tprs, third_fprs)


if __name__ == '__main__':
    main()
