from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# MNIST
digits = datasets.load_digits()

X = digits.data
y = digits.target

# just 0 and 1's
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)

linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)

linear_acc = accuracy_score(y_test, y_pred_linear)
print(f"Accuracy with Linear Kernel: {linear_acc:.2f}")

rbf_svm = SVC(kernel='rbf', gamma='auto')
rbf_svm.fit(X_train, y_train)

y_pred_rbf = rbf_svm.predict(X_test)

rbf_acc = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy with RBF Kernel: {rbf_acc:.2f}")
