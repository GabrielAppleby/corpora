"""
Gabriel Appleby
Working With Corpora
Logistic Regression

This file performs the actual logistic classification
"""

import numpy as np

from constants import BALANCED_FINAL_DATA, UNBALANCED_FINAL_DATA
from logistic_classifier import LogisticClassifier


def main() -> None:
    """
    Performs classificaiton and scoring
    :return: None.
    """
    unbalanced_score = classify_and_score(*load_data(UNBALANCED_FINAL_DATA))
    print("Unbalanced data score: {}.".format(unbalanced_score))

    balanced_score = classify_and_score(*load_data(BALANCED_FINAL_DATA))
    print("Balanced data score: {}.".format(balanced_score))


def load_data(file_path):
    """
    Loads the npz at the given filepath
    :param file_path: The path of the npz to load
    :return: X_train, X_test, y_train, y_test
    """
    unbalanced_data = np.load(file_path)
    X_train = unbalanced_data["X_train"]
    X_test = unbalanced_data["X_test"]
    y_train = unbalanced_data["y_train"]
    y_test = unbalanced_data["y_test"]

    return X_train, X_test, y_train, y_test


def classify_and_score(X_train, X_test, y_train, y_test):
    """
    Classifies and scores the data
    :param X_train: The training feature matrix
    :param X_test: The test feature matrix
    :param y_train: The training labels
    :param y_test: The test labels
    :return:
    """
    clf: LogisticClassifier = LogisticClassifier()
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


if __name__ == "__main__":
    main()
