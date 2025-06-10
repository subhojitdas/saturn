#shared copied from here - https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

import numpy as np
import pickle
import gym

H = 200
batch_size = 10
lr = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99
resume = False
render = False

D = 80 * 80

if resume:
    model = pickle.load(open('./models/saved.pkl', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def image_pre_processor(Im):
    """pre process 210*160*3 uint8 frames into 80*80 1D vector"""
    Im = Im[35:195]
    Im = Im[::2, ::2, 0]
    Im[Im == 144] = 0
    Im[Im == 109] = 0
    Im[Im != 0] = 1
    return Im.astype(np,float).ravel()



