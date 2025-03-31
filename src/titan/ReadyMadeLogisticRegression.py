from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



from src.titan.heart_disease_predictor import LogisticRegressionHeartDisease

logistic_regression_heart_disease = LogisticRegressionHeartDisease()
df = logistic_regression_heart_disease.load_data_set()

X, y = logistic_regression_heart_disease.prepare_dataset(df)
X_train, X_test, y_train, y_test = logistic_regression_heart_disease.split_dataset(X, y, True)

# Polynomial boosting did not help
# poly = PolynomialFeatures(degree=2)
# X_train = poly.fit_transform(X_train)
# X_test = poly.transform(X_test)
#
# model = LogisticRegression(C=1/0.1, penalty='l2', max_iter=1000)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# accuracy = np.mean(predictions == y_test)
# print(f"Scikit-learn Accuracy: {accuracy}")


# model = RandomForestClassifier(n_estimators=10, random_state=18)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# accuracy = np.mean(predictions == y_test)
# print(f"Random Forest Accuracy: {accuracy}")
#
# model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=18)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# accuracy = np.mean(predictions == y_test)
# print(f"XGBoost Accuracy: {accuracy}")
#
# from sklearn.feature_selection import SelectKBest, f_classif
#
# selector = SelectKBest(f_classif, k=10)  # Select top 10 features
# X_train_selected = selector.fit_transform(X_train, y_train)
# X_test_selected = selector.transform(X_test)
#
# # Train the model with reduced features
# model.fit(X_train_selected, y_train)
# predictions = model.predict(X_test_selected)
#
# accuracy = np.mean(predictions == y_test)
# print(f"Accuracy after feature selection: {accuracy}")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=18)
#
# # Fit the model on training data
# svm_model.fit(X_train, y_train)
#
# # Predict on test set
# predictions = svm_model.predict(X_test)
#
# # Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f"SVM with RBF Kernel Accuracy: {accuracy}")

param_grid = {
    'C': [0.5, 1, 5, 10, 50],
    'gamma': [0.01, 0.1, 'scale', 'auto', 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(random_state=18), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Params: {grid_search.best_params_}")

predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Optimized SVM Accuracy: {accuracy}")

