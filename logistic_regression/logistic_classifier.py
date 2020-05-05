import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class LogisticClassifier(BaseEstimator, ClassifierMixin):
    """
    A homemade logistic classifier that uses gradient descent.
    """

    def __init__(self, lr=.002, steps=10):
        """
        Initializes the classifier.
        :param lr: The lr to use when adjusting weights during gradient descent.
        :param steps: The number of steps to take during the descent.
        """
        self.__weights = None
        self.__lr = lr
        self.__steps = steps
        super().__init__()

    def fit(self, X, y):
        """
        Fits the weights to the data. Subsequent calls reset the weight matrix each time.
        :param X: The (n X f) matrix with n samples and f features.
        :param y: The (n X 1) matrix of labels.
        :return: self, the fit classifier.
        """
        # Add intercept
        X = self.__add_intercept(X)
        num_samples, num_features = X.shape

        # Reshape y if it is not in the appropriate shape.
        y = y.reshape((-1, 1))

        # Randomly generates the weights.
        self.__weights = np.random.random_sample((num_features, 1))

        # Train for self.__steps
        for _ in range(self.__steps):
            y_pred = self.__predict(X)
            self.__weights += np.sum(self.__lr * X * (y - y_pred), axis=0).reshape(-1, 1)

        return self

    def predict(self, X):
        """
        Predicts based upon the trained weights. Do not call before calling fit (sorry). Throws
        an exception if fit has not been called.
        :param X: The (n X f) matrix with n samples and f features to predict the labels of.
        :return: The predictions.
        """
        assert(self.__weights is not None)

        X = self.__add_intercept(X)
        y_pred = self.__predict(X)

        return self.__round_preds(y_pred)

    def score(self, X, y, sample_weight=None):
        """
        Report the accuracy score of the classifier.
        :param X: The (n X f) matrix with n samples and f features to predict the labels of.
        :param y: The (n X 1) matrix of true labels from which to compute the accuracy score.
        :param sample_weight: This parameter is ignored. Simply here to adhere to the interface.
        :return: The accuracy score.
        """
        assert (self.__weights is not None)

        X = self.__add_intercept(X)
        y_pred = self.__predict(X)
        y_pred = self.__round_preds(y_pred)

        return accuracy_score(y, y_pred)

    # "Private" functions.

    def __predict(self, X):
        # Numerically stable
        return expit(np.matmul(X, self.__weights))

    @staticmethod
    def __loss(y_true, y_pred):
        log_y_pred = np.log(y_pred)
        log_one_minus_y_pred = np.log(1-y_pred)
        one_minus_y_true = 1 - y_true

        return -np.sum(y_true * log_y_pred + one_minus_y_true * log_one_minus_y_pred)

    @staticmethod
    def __add_intercept(X):
        num_samples, num_features = X.shape

        return np.append(X, np.ones((num_samples, 1), dtype=np.float64), axis=1)

    @staticmethod
    def __round_preds(y_pred):
        y_pred[y_pred >= .5] = 1
        y_pred[y_pred < .5] = 0

        return y_pred