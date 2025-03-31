import pandas as pd
import kagglehub
from sklearn.preprocessing import LabelEncoder

from src.titan.logistic_regressor_sub import LogisticRegressionSub


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

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = model.accuracy(y_test, y_pred)

print(f"Accuracy: {accuracy}")


