from convnet.data_loader import CIFAR_10_DataLoader
from convnet.struct.conv_layer import SimpleConvNet, MaxPoolingConvNet
from convnet.struct.layers import ReLU, Linear
from convnet.struct.loss import SoftmaxCrossEntropyLoss

import numpy as np


class GolemCNN:

    def __init__(self):
        # self.conv1 = SimpleConvNet(kernel_size=6, depth=3, spatial_dim=5)
        # self.relu1 = ReLU()
        # self.maxpool1 = MaxPoolingConvNet(spatial_dim=2, stride=2)
        #
        # self.conv2 = SimpleConvNet(kernel_size=10, depth=6, spatial_dim=5)
        # self.relu2 = ReLU()
        # self.maxpool2 = MaxPoolingConvNet(spatial_dim=2, stride=2)
        #
        fc_fan_in = 10 * 5 * 5
        num_classes = 10
        # self.fc1 = Linear(fc_fan_in, num_classes)
        self.layers = [
            SimpleConvNet(kernel_size=6, depth=3, spatial_dim=5),
            ReLU(),
            MaxPoolingConvNet(spatial_dim=2, stride=2),
            SimpleConvNet(kernel_size=10, depth=6, spatial_dim=5),
            ReLU(),
            MaxPoolingConvNet(spatial_dim=2, stride=2),
            Linear(fc_fan_in, num_classes),
        ]

    def forward(self, x):
        # out = self.conv1.forward(x)
        # out = self.relu1.forward(out)
        # out = self.maxpool1.forward(out)
        #
        # out = self.conv2.forward(out)
        # out = self.relu2.forward(out)
        # out = self.maxpool2.forward(out)
        #
        # out = out.reshape(out.shape[0], -1) # flatten
        # out = self.fc1.forward(out)

        for layer in self.layers:
            if layer == self.layers[-1]:
                x = x.reshape(x.shape[0], -1)
            x = layer.forward(x)
        out = x
        return out

    def backward(self, dout):
        # dout = self.fc1.backward(dout)
        # dout = dout.reshape(self.maxpool2.pooled_out.shape)
        # dout = self.maxpool2.backward(dout)
        # dout = self.relu2.backward(dout)
        # dout = self.conv2.backward(dout)
        # dout = self.maxpool1.backward(dout)
        # dout = self.relu1.backward(dout)
        # dout = self.conv1.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if layer == self.layers[-1]:
                dout = dout.reshape(self.layers[-2].pooled_out.shape)
        return dout

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, SimpleConvNet) or isinstance(layer, Linear):
                layer.update_parameters(learning_rate)


def split_loss(model, split):
    x, y = {
        'train': (Xtrain, ytrain),
        'dev': (Xdev, ydev),
        'test': (Xtest, ytest),
    }[split]
    x = model.forward(x)
    logits = x
    loss = loss_criteria.forward(logits, y)
    print(f"{split} => loss: {loss:.4f}")

def accuracy(x, labels, model):
    for layer in model.layers:
        # if isinstance(layer, BatchNorm1d):
        #     layer.train = False
        x = layer.forward(x)
    logits = x
    probs = loss_criteria.softmax_numpy(logits)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == labels)

file_directory = '/Users/subhojit/Downloads/cifar-10-batches-py'
cdl = CIFAR_10_DataLoader()
xtrain_data, ytrain_data, Xtest, ytest = cdl.load_cifar_10_dataset(file_directory)
xtrain_data = xtrain_data.reshape(-1, 32, 32, 3)
Xtest = Xtest.reshape(-1, 32, 32, 3)

xtrain_data = xtrain_data.astype('float32') / 255.0
Xtest = Xtest.astype('float32') / 255.0

# np.random.shuffle(Xtrain)
n1 = int(0.8 * len(xtrain_data))
Xtrain = xtrain_data[:n1]
ytrain = ytrain_data[:n1]
Xdev = xtrain_data[n1:]
ydev = ytrain_data[n1:]

num_classes = len(set(ytrain))


max_iterations = 1000
batch_size = 128
lossi = []
Hs = []

model = GolemCNN()
loss_criteria = SoftmaxCrossEntropyLoss()

for i in range(max_iterations):

    #mini batch
    ix = np.random.randint(0, Xtrain.shape[0], (batch_size,))
    Xb, Yb = Xtrain[ix], ytrain[ix]

    logits = model.forward(Xb)
    loss = loss_criteria.forward(logits, Yb)
    lossi.append(loss)

    logits_grad = loss_criteria.backward()
    model.backward(logits_grad)

    lr = 0.1
    model.update_parameters(lr)

    if i % 10 == 0:
        print(f"loss: {loss}")
        break


split_loss(model, 'train')
split_loss(model, 'dev')
split_loss(model, 'test')

accuracy = accuracy(Xtest, ytest, model)
print(f"accuracy: {accuracy}")