import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionHeartDisease:

    def __init__(self):
        self.theta = None
        self.threshold = 0.5

    def download_data(self):
        path = kagglehub.dataset_download("alexisbcook/cleveland-clinic-foundation-heart-disease")
        print("Path to dataset files:", path)
        path = '/Users/subhojit/.cache/kagglehub/datasets/alexisbcook/cleveland-clinic-foundation-heart-disease/versions/1'

    def load_data_set(self):
        file_path = '../dataset/heart.csv'
        df = pd.read_csv(file_path)
        # print("Dataset shape:", df.shape)
        # print(df.head())
        return df

    def prepare_dataset(self, df):
        print(df['thal'].unique())
        print(df.shape)
        df['thal'] = df['thal'].replace('1', np.nan)
        df['thal'] = df['thal'].replace('2', np.nan)
        df.dropna(subset=['thal'], inplace=True)
        print(df.shape)

        # convert thal to one hot encoding
        df = pd.get_dummies(df, columns=['thal'], drop_first=True)
        df[['thal_normal', 'thal_reversible']] = df[['thal_normal', 'thal_reversible']].astype(int)
        X = df.drop('target', axis=1)
        y = df['target']
        return X, y

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

            gradient = (1/feature_dimention) * (x_with_bias.T @ (h - y_train))
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


#### ************************************************** ######
'''
1. It load the data to a dataframe
2. Prepares the data
    1. There is one feature called 'thal' which has some enum values ['fixed' 'normal' 'reversible' '1' '2']
    2. It drop '1' and '2' valued rows
    3. It does one-hot encoding for the other values. 
    4. It created two new rows 'thal_fixed' and 'thal_reversible' and dropped 'normal' to make the feature independent
    5. It converts the value of 'thal_normal' and 'thal_reversible' to be 0 or 1
3.  It split the data into train and test sets
4. It does z-score normalization on X_train and X_test
5. It then tries to fit the model with X_train and y_train with logistic  regression model. 
    h(x) = g(theta.T dot X) = 1 / (1 + np.exp(-theta.T dot X))
6. It gets a linear decision boundary among the training set
7. Accuracy test on test set
'''
#### ************************************************** ######
model = LogisticRegressionHeartDisease()
df = model.load_data_set()
X, y = model.prepare_dataset(df)
X_train, X_test, y_train, y_test = model.split_dataset(X, y, True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = model.accuracy(y_test, predictions)

print(f"Accuracy: {accuracy}")






