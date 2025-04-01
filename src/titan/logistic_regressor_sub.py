import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionSub:

    def __init__(self):
        self.theta = None
        self.threshold = 0.5

    def split_dataset(self, X, y, with_z_score_normalization=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

        # z-score normalization
        if with_z_score_normalization:
            scalar = StandardScaler()
            X_train = scalar.fit_transform(X_train)
            X_test = scalar.transform(X_test)

        return X_train, X_test, y_train, y_test

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X_train, y_train):
        feature_sample, feature_dimention = X_train.shape
        theta = np.zeros(feature_dimention + 1)
        x_with_bias = np.c_[np.ones((feature_sample, 1)), X_train]

        for i in range(1000):
            z = x_with_bias @ theta
            h = self.sigmoid(z)

            gradient = (1/feature_sample) * (x_with_bias.T @ (h - y_train))
            theta -= gradient

        self.theta = theta

    def fit_with_regularization(self, X_train, y_train, lambda_regularization=0.1):
        feature_sample, feature_dimention = X_train.shape
        theta = np.zeros(feature_dimention + 1)
        x_with_bias = np.c_[np.ones((feature_sample, 1)), X_train]

        for i in range(1000):
            z = x_with_bias @ theta
            h = self.sigmoid(z)

            gradient = (1/feature_sample) * (x_with_bias.T @ (h - y_train))
            # Add L2 penalty term (ignore bias term by excluding theta[0])
            gradient[1:] += (lambda_regularization / feature_sample) * theta[1:]
            theta -= gradient

        self.theta = theta

    def predict_probability(self, X):
        test_sample = X.shape[0]
        x_with_bias = np.c_[np.ones((test_sample, 1)), X]
        z = x_with_bias @ self.theta
        h = self.sigmoid(z)
        return h

    def predict(self, X):
        probabilities = self.predict_probability(X)
        return probabilities >= self.threshold

    def accuracy(self, y, predictions):
        return np.mean(y == predictions)



