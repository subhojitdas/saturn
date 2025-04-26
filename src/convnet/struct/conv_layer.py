import numpy as np

class SimpleConvNet:

    def __init__(self, kernel_size, depth, spatial_dim=5, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.spatial_dim = spatial_dim
        self.depth = depth
        self.stride = stride
        self.padding = padding
        # for i in range(kernel_size):
        #     F = np.random.randn(self.spatial_dim, self.spatial_dim, self.depth)
        #     self.kernel_list.append(F)
        # vectorized
        self.kernel_list = np.random.randn(self.kernel_size, self.spatial_dim, self.spatial_dim, self.depth)
        self.biases = np.zeros(self.kernel_size)

    def forward(self, x):
        self.x = x # saving for backward
        # x (W1, H1, D1)
        N, W1, H1, D1 = x.shape

        out_h = (W1 - self.spatial_dim) // self.stride + 1
        out_w = (H1 - self.spatial_dim) // self.stride + 1
        activation_map = np.zeros((N, out_h, out_w, self.kernel_size))

        for n in range(N):
            for k in range(self.kernel_size):
                kernel = self.kernel_list[k]
                for i in range(out_h):
                    for j in range(out_w):
                        patch = x[n,
                                    i*self.stride:i*self.stride+self.spatial_dim,
                                    j*self.stride:j*self.stride+self.spatial_dim, :]
                        activation_map[n, i, j, k] = np.sum(patch * kernel) + self.biases[k]

        return activation_map

    def backward(self, dout):
        x = self.x
        N, W, H, D = x.shape
        out_h = (W - self.spatial_dim) // self.stride + 1
        out_w = (H - self.spatial_dim) // self.stride + 1

        dx = np.zeros_like(x)
        dW = np.zeros((self.kernel_size, self.spatial_dim, self.spatial_dim, self.depth))
        db = np.zeros(self.kernel_size)
        for n in range(N):
            for k in range(self.kernel_size):
                kernel = self.kernel_list[k]
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x[n, i_start:i_start+self.spatial_dim, j_start:j_start+self.spatial_dim, :]
                        db[k] += dout[n, i, j, k]
                        dW[k] += patch * dout[n, i, j, k]
                        dx[n, i_start:i_start+self.spatial_dim, j_start:j_start+self.spatial_dim, :] += kernel * dout[n, i, j, k]

        self.dW = dW
        self.db = db
        return dx

    def update_parameters(self, lr):
        self.kernel_list += -lr * self.dW
        self.biases += -lr * self.db


class MaxPoolingConvNet:
    def __init__(self, spatial_dim=2, stride=2):
        self.spatial_dim = spatial_dim
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, W1, H1, D1 = x.shape
        out_h = (W1 - self.spatial_dim) // self.stride + 1
        out_w = (H1 - self.spatial_dim) // self.stride + 1
        self.pooled_out = np.zeros((N, out_h, out_w, D1))

        for i in range(out_h):
            for j in range(out_w):
                i_start = i * self.stride
                j_start = j * self.stride
                self.pooled_out[:, i, j, :] = np.max(x[:, i_start:i_start+self.spatial_dim, j_start:j_start+self.spatial_dim, :], axis=(1, 2))
        return self.pooled_out

    def backward(self, dout):
        x = self.x
        out_h, out_w = dout.shape[1], dout.shape[2]

        dx = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                i_start = i * self.stride
                j_start = j * self.stride
                patch = x[:, i_start:i_start+self.spatial_dim, j_start:j_start+self.spatial_dim, :]
                match_patch = np.max(patch, axis=(1, 2), keepdims=True)
                mask = patch == match_patch
                dx[:, i_start:i_start+self.spatial_dim, j_start:j_start+self.spatial_dim, :] += mask * dout[:, i:i+1, j:j+1, :]

        return dx





