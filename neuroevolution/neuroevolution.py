from audioop import avg
from os import popen
from neuroevolver import Neuroevolver
from mlp_from_scratch.nn_final import Network
import matplotlib.pyplot as plt
import numpy as np
import random
from mlp_from_scratch.mnist import load_data_wrapper
import random

def fitness(net, image, result):
    image.reshape(784)
    out = net.feed_forward(image)
    return np.linalg.norm((out-result)**2)

POP_SIZE = 200
MUTATION_RATE = 0.001

nets = [Neuroevolver([784,16,16,10]) for i in range(POP_SIZE)]

x_train, y_train, x_test = load_data_wrapper()

pop_fitness = sum(net.fitness for net in nets) / len(nets)

best_net = None

while pop_fitness >= 0.1 or pop_fitness < 0:
    print(pop_fitness)
    min_net = None
    min_fit = None
    for net in nets:
        avg_fit = 0

        for i in range(1000):
            image, result = x_train[random.randint(0,len(x_train)-1)]
            avg_fit += net.evaluate(image, result, fitness)

        avg_fit /= 1000

        if min_fit is None or avg_fit < min_fit:
                min_fit = avg_fit
                min_net = net
                net.fitness = avg_fit
                pop_fitness = avg_fit
                best_net = min_net
    
    nets = [min_net for i in range(POP_SIZE)]
    for net in nets:
        net.mutate(MUTATION_RATE)
    

test_image, test_label = x_test[random.randint(0,len(x_test))]
show_image = np.reshape(test_image,(28,28))
plt.imshow(show_image)
plt.show()
out = best_net.feed_forward(test_image)
new = []
for i in range(len(out)):
    new.append(out[i][0])
guess = new.index(max(new))
print("\n\nGuess: " + str(guess))
print(f"Certainty: {round(max(new)*100,2)}%")
        