import numpy as np
import pickle

class KNearestNeighbour:

    def __init__(self, X, y):
        self.Xtrain = X
        self.ytrain = y

    def predict(self, X, k):
        num_test = X.shape[0]
        ypred = np.zeros(num_test, dtype=self.ytrain.dtype)

        for i in range(num_test):
            distance = np.sum(np.abs(self.Xtrain - X[i, :]), axis=1)
            k_min_indices = np.argpartition(distance, k)
            k_min_values = self.ytrain[k_min_indices]
            bincounts = np.bincount(k_min_values)
            ypred[i] = np.argmax(bincounts)
            print(f'predicted picture {i}')

        return ypred


def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        # cifar-10 stores raw vector 3072 bytes. reshaping to 3 channel * 32 height * 32 width
        images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return data, np.array(labels)


def load_cifar_10_dataset(file_directory):

    train_images = []
    train_labels = []
    for i in range(1, 6, 1):
        images, labels = load_cifar_batch(f"{file_directory}/data_batch_{i}")
        train_images.append(images)
        train_labels.append(labels)
    Xtrain = np.concatenate(train_images)
    Ytrain = np.concatenate(train_labels)

    Xtest, Ytest = load_cifar_batch(f"{file_directory}/test_batch")

    return Xtrain, Ytrain, Xtest, Ytest

def calculate_accuracy(ypred, ytest):
    num_test = ytest.shape[0]
    mismatch_count = 0
    for i in range(num_test):
        if ypred[i] != ytest[i]:
            mismatch_count += 1
    return (num_test - mismatch_count) / num_test



Xtrain, ytrain, Xtest, ytest = load_cifar_10_dataset('/Users/subhojit/Downloads/cifar-10-batches-py')

nearest_neighbor = KNearestNeighbour(Xtrain, ytrain)
print(f"Total : {ytest.shape[0]}")
ypred = nearest_neighbor.predict(Xtest, 3)
accuracy = calculate_accuracy(ypred, ytest)
print(f"Accuracy: {accuracy}")



