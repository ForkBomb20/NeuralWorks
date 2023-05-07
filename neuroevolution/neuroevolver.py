from mlp_from_scratch.nn_final import Network
import random
import numpy as np

class Neuroevolver(Network):
    def __init__(self, sizes):
        super().__init__(sizes)
        self.fitness = -1

    def evaluate(self, image, result, f):
        return f(self, image, result)

    def mutate(self, rate):
        for i in range(len(self.weights)):
            for r in range(len(self.weights[i])):
                for c in range(len(self.weights[i][r])):
                    if random.random() < rate:
                        self.weights[i][r][c] = random.random() * 2 - 1
                        self.bias[i][r] + random.random() * 2 - 1

