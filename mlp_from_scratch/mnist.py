import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('./data/digits/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    # p = u.load()
    # training_data, validation_data, test_data = pickle.load(f)
    x_train, y_train, x_test = u.load()
    f.close()
    return (x_train, y_train, x_test)

def load_data_wrapper():
    x_train, y_train, x_test = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in x_train[0]]
    training_results = [vectorized_result(y) for y in x_train[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in y_train[0]]
    validation_data = list(zip(validation_inputs, y_train[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in x_test[0]]
    test_data = list(zip(test_inputs, x_test[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e