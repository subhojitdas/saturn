import pandas as pd
import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

from src.titan.logistic_regressor_sub import LogisticRegressionSub
from src.titan.random_forest import RandomForestClassifierSub


def load_uci_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    columns = ["A" + str(i) for i in range(1, 15)] + ["Target"]
    data = pd.read_csv(url, header=None, names=columns, delimiter=" ")
    print(data.shape)

def load_data_kaggle():
    path = kagglehub.dataset_download("rikdifos/credit-card-approval-prediction")
    print("Path to dataset files:", path)


app_data = pd.read_csv("../dataset/application_record.csv")
credit_data = pd.read_csv("../dataset/credit_record.csv")
print('App data shape: ', app_data.shape)
print('Credit record shape: ', credit_data.shape)

defaulter_id = credit_data[credit_data['STATUS'].isin(['2', '3', '4', '5'])]["ID"].unique()

target_data = pd.DataFrame(credit_data["ID"], columns=["ID"])
target_data["target"] = target_data["ID"].apply(lambda x: 0 if x in defaulter_id else 1)

print(target_data.value_counts())

# Data seems to be very imbalanced in this dataset. If we consider status >= 3 is defaulter ,
# then I am getting only 331 samples for defaulter out of 45000+ samples. 99.3-0.7% distribution ?
# down-sampling

merged_data = pd.merge(app_data, target_data, on="ID", how="inner")
majority_class = merged_data[merged_data["target"] == 1]
minority_class = merged_data[merged_data["target"] == 0]

drop_fraction = 0.91
majority_class_reduced = majority_class.sample(frac=(1 - drop_fraction), random_state=18)
balanced_data = pd.concat([majority_class_reduced, minority_class], axis=0)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(balanced_data["target"].value_counts())

cat_columns = balanced_data.select_dtypes(include=["object"]).columns
print(cat_columns)

label_encoder = LabelEncoder()

for col in cat_columns:
    balanced_data[col] = label_encoder.fit_transform(balanced_data[col])

X = balanced_data.drop('target', axis=1)
y = balanced_data['target']

model = LogisticRegressionSub()
X_train, X_test, y_train, y_test = model.split_dataset(X, y, with_z_score_normalization=False)

# z-score normalization only for the numeric features
numeric_columns = [col for col in X_train.columns if col not in cat_columns]
print(numeric_columns)

scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Own LR
# model.fit_with_regularization(X_train, y_train, lambda_regularization=0)
# y_pred = model.predict(X_test)
# accuracy = model.accuracy(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
#
# # sklearn LR
# model = LogisticRegression(C=1/0.1, penalty='l2', max_iter=1000)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# accuracy = np.mean(predictions == y_test)
# print(f"Scikit-learn Accuracy: {accuracy}")
#
# ## Random Forest
# model = RandomForestClassifier(n_estimators=10, random_state=18)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# accuracy = np.mean(predictions == y_test)
# print(f"Random Forest Accuracy: {accuracy}")
import cProfile

sub_model = RandomForestClassifierSub(n_estimators=1)

profiler = cProfile.Profile()
profiler.enable()

cProfile.run("sub_model.fit(X_train, y_train)", filename="/tmp/random_forest_model")

profiler.disable()
profiler.dump_stats("/tmp/profile_results.prof")

# sub_model.fit(X_train, y_train)
predictions = sub_model.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f"Own Random Forest Accuracy: {accuracy}")