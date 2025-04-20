import pickle
import numpy as np

class CIFAR_10_DataLoader:

    def load_cifar_batch(self, file_path):
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']
            # cifar-10 stores raw vector 3072 bytes. reshaping to 3 channel * 32 height * 32 width
            images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return data, np.array(labels)

    def load_cifar_10_dataset(self, file_directory):
        train_images = []
        train_labels = []
        for i in range(1, 6, 1):
            images, labels = self.load_cifar_batch(f"{file_directory}/data_batch_{i}")
            train_images.append(images)
            train_labels.append(labels)
        Xtrain = np.concatenate(train_images)
        Ytrain = np.concatenate(train_labels)

        Xtest, Ytest =self.load_cifar_batch(f"{file_directory}/test_batch")

        return Xtrain, Ytrain, Xtest, Ytest