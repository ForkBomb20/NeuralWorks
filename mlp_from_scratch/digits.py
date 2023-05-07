from nn_final import Network
import matplotlib.pyplot as plt
import numpy as np
from mlp_from_scratch.mnist import load_data_wrapper
import random
from tqdm import tqdm
import sys

net = Network([784,16,16,10])
net.weights = np.load("./data/digits/weights.npy",allow_pickle=True)
net.bias = np.load("./data/digits/bias.npy",allow_pickle=True)

x_train, y_train, x_test = load_data_wrapper()

with tqdm(total=len(x_train),desc=f"Training Progress") as pbar:
    for i in range(len(x_train)):
        image, result = x_train[i]
        w_grad, b_grad = net.back_prop(image, result)
        net.weights = np.subtract(net.weights,w_grad)
        net.bias = np.subtract(net.bias,b_grad)
        if i % 500 == 0:
            if i == 0:
                print("\n\n")
            pbar.update(500)
            pbar.set_postfix_str(s=f"Training Cost: {round(net.av_cost,3)}",refresh=True)

test_image, test_label = x_test[random.randint(0,len(x_test))]
show_image = np.reshape(test_image,(28,28))
plt.imshow(show_image)
plt.show()
out = net.feed_forward(test_image)
new = []
for i in range(len(out)):
    new.append(out[i][0])
guess = new.index(max(new))
print("\n\nGuess: " + str(guess))
print(f"Certainty: {round(max(new)*100,2)}%")
np.save("./data/digits/weights",net.weights,allow_pickle=True)
np.save("./data/digits/bias",net.bias,allow_pickle=True)