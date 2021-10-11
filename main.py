# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import mnist
import tensorflow.keras as keras
import numpy as np


def combine_data(image_datasets, label_datasets):
    # Flatten the images and add 1 to first element.
    image_vector_size = 28 * 28

    image_datasets = image_datasets.reshape(image_datasets.shape[0], image_vector_size)
    image_datasets = np.insert(image_datasets, 0, 255, axis=1) / 255
    label_datasets = keras.utils.to_categorical(label_datasets, 10)

    yield from ((image, label) for image, label in zip(image_datasets, label_datasets))


class Network(object):
    def __init__(self, sizes, activation_func, activation_func_derivative, initial_weights_func, use_softmax=True):

        # Setup Configurations.
        self.sizes = sizes
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
        self.use_softmax = use_softmax

        # Initialize weights.
        assert (len(self.sizes) >= 3)
        self.w = [initial_weights_func(n_out, n_in)
                  for n_out, n_in in zip(self.sizes[1:], self.sizes[:-1])]

        # Initialize input and output after process activation function of layers.
        self.z = [np.zeros((s, 1)) if i > 0 else None for i, s in enumerate(self.sizes)]
        self.y = [np.zeros((s, 1)) for s in self.sizes]

    def feedforward(self, x):
        # Store the output of the input layer as data from datasets.
        self.y[0] = np.array(x, dtype=float)

        # Compute input/output for hidden layers.
        n_hidden = len(self.w) - 1
        for i in range(n_hidden):
            # First compute weighted sum of inputs (outputs from previous layer).
            # Then run the activation function on these values to produces the outputs for this hidden layer.
            # Store input/output values for back propagation purposes.
            self.z[i + 1] = np.dot(self.w[i], self.y[i])
            self.y[i + 1] = self.activation_func(self.z[i + 1])

        # With Output layer: Use Only softmax or sigmoid activation functions.
        self.z[-1] = np.dot(self.w[-1], self.y[-2])
        if self.use_softmax:
            self.y[-1] = softmax(self.z[-1])
        else:
            self.y[-1] = np.divide(1.0, np.add(1.0, np.exp(np.multiply(-1.0, self.z[-1]))))

        # Return the output layer values.
        return self.y[-1]

    def update(self, batch, eta):
        # Extract image data from batch.
        x = [batch[i][0] for i in range(len(batch))]
        y_target = [batch[i][1] for i in range(len(batch))]
        sum_grad_w = [np.zeros(w.shape) for w in self.w]
        for i in range(len(x)):
            # Feed Forward.
            self.feedforward(x[i])

            # Compute and accumulate error gradients.
            grad_w = self._backpropagate(y_target[i])
            for j in range(len(grad_w)):
                sum_grad_w[j] += grad_w[j]

        # Update weights before experiencing next batch.
        for i in range(len(self.w)):
            self.w[i] = np.subtract(self.w[i], np.multiply(eta, sum_grad_w[i]))

    def _backpropagate(self, t):
        # Compute deltas for the output layer. These deltas become input to update weights of the hidden layers.
        # Assumes activation function in the output layer is softmax or sigmoid.
        delta = np.asmatrix(np.subtract(self.y[-1], t))

        # Compute deltas and gradients for hidden layers.
        grad = [np.zeros(w.shape) for w in self.w]
        for i in range(len(self.w) - 1, 0, -1):
            # Compute the gradients for this weight layer using deltas from previous layer.
            grad[i] = np.dot(np.transpose(delta), np.asmatrix(self.y[i]))

            # Compute deltas for this layer using deltas from the layer directly above.
            # Compute the activation function and its derivative for the inputs to this layer.
            dy_dz = np.asmatrix(self.activation_func_derivative(self.z[i]))
            h_sums = np.dot(delta, self.w[i])
            np.resize(delta, (dy_dz.shape[0] * h_sums.shape[0], 1))
            delta = np.multiply(dy_dz, h_sums)

        # Compute gradients for input layer (using deltas from first hidden layer).
        grad[0] = np.dot(np.transpose(delta), np.asmatrix(self.y[0]))

        return grad


def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def sigmoid(z):
    return np.divide(1.0, np.add(1.0, np.exp(-z)))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return np.multiply(s, np.subtract(1.0, s))


def sigmoid_initial_weights(n_out, n_in):
    return 4.0 * np.random.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),
                                   high=np.sqrt(6.0 / (n_in + n_out)),
                                   size=(n_out, n_in))


def train_network(net, data, epoch_count, batch_size, eta, error_rate_func=None):
    n = len(data)
    for e in range(epoch_count):
        np.random.shuffle(data)
        for k in range(0, n, batch_size):
            mini_batch = data[k:k + batch_size]
            net.update(mini_batch, eta)
            print("\rEpoch %02d, %05d instances" % (e + 1, k + batch_size), end="")
        print()
        if error_rate_func:
            error_rate = error_rate_func(net)
            print("Epoch %02d, error rate = %.2f" % (e + 1, error_rate * 100))


def get_error_rate(net, error_data):
    assert len(error_data) > 0
    error_count = 0.0
    for image, label in error_data:
        y_hat = net.feedforward(image)
        if np.argmax(y_hat, axis=0) != np.argmax(label, axis=0):
            error_count += 1.0

    return error_count / len(test_data)


def network_sizes(input_layer_size, output_layer_size, hidden_layer_sizes):
    sizes = [input_layer_size]
    for hidden_layer_size in hidden_layer_sizes:
        sizes.append(hidden_layer_size)
    sizes.append(output_layer_size)
    return sizes


if __name__ == "__main__":
    # Setup train and test splits
    (image_train, label_train), (image_test, label_test) = mnist.load_data()
    training_data = list(combine_data(image_train, label_train))
    test_data = list(combine_data(image_test, label_test))

    # Make sure to use the same seed so that can compare results between runs.
    np.random.seed(1234)

    # Configurations.
    EPOCH_COUNT = 10
    BATCH_SIZE = 20
    LEARNING_RATE = 0.1
    INPUT_LAYER_SIZE = 785
    OUTPUT_LAYER_SIZE = 10
    HIDDEN_LAYER_SIZES = [100, 100]
    USE_SOFTMAX = True
    ACTIVATION_FUNC = sigmoid
    ACTIVATION_FUNC_DERIVATIVE = sigmoid_derivative
    INITIAL_WEIGHTS_FUNC = sigmoid_initial_weights

    # Process Feed Forward Neuron Network Training
    network = Network(sizes=network_sizes(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES),
                      activation_func=ACTIVATION_FUNC,
                      activation_func_derivative=ACTIVATION_FUNC_DERIVATIVE,
                      initial_weights_func=INITIAL_WEIGHTS_FUNC,
                      use_softmax=USE_SOFTMAX)
    train_network(network, training_data, epoch_count=EPOCH_COUNT, batch_size=BATCH_SIZE, eta=LEARNING_RATE,
                  error_rate_func=lambda n: get_error_rate(n, test_data))
