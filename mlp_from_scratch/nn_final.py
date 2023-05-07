import numpy as np
import random
import matplotlib.pyplot as plt

from mnist import load_data_wrapper

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))


class Network:
    def __init__(self,sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(r,c) for c,r in zip(sizes[::],sizes[1:])]
        self.bias = [np.random.randn(size,1) for size in sizes[1:]]
        self.zs = []
        self.activations = []
        self.av_cost = 0

    def feed_forward(self,activation):
        self.activations = [activation]
        self.zs = [activation]
        for w,b in (zip(self.weights,self.bias)):
            self.zs.append(np.dot(w,self.activations[len(self.activations)-1]) + b)
            self.activations.append(sigmoid(np.dot(w,self.activations[len(self.activations)-1]) + b))

        return self.activations[self.num_layers-1]

    def back_prop(self,activation,y):
        a = self.feed_forward(activation)
        w_grad = [np.zeros(w.shape) for w in self.weights]
        b_grad = [np.zeros(b.shape) for b in self.bias]
        delta = (a-y) * dsigmoid(self.zs[-1])
        b_grad[-1] = delta
        w_grad[-1] = 0.30 * np.dot(delta,np.transpose(self.activations[-2]))

        self.av_cost = np.linalg.norm((a-y)**2)

        for l in range(2,self.num_layers):
            w = self.weights[-l+1]
            z = self.zs[-l]
            a = self.activations[-l-1]

            delta = np.dot(np.transpose(w),delta) * dsigmoid(z) 
            b_grad[-l] = delta
            w_grad[-l] = 0.30 * np.multiply(delta,np.transpose(a))
            

        return(w_grad,b_grad)

       



    


