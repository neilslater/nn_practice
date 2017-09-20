import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import math

def get_data():
    X, y = sklearn.datasets.make_moons(600, noise=0.30)
    y = y.reshape([600,1])
    X_train = X[:400]; y_train = y[:400]
    X_cv = X[400:500]; y_cv = y[400:500]
    X_test = X[500:]; y_test = y[500:]

    # Numpy model works with each column as an example, so that np.dot can be broadcast on a mini-batch
    return (X_train.T, y_train.T, X_cv.T, y_cv.T, X_test.T, y_test.T)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_gradient(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_gradient(y):
    return 1.0 - y**2

def relu(x):
    return (x > 0).astype('float') * x

def relu_gradient(y):
    return (y > 0).astype('float')

class NN():
    def __init__(self, layer_sizes, hidden_activation='relu', l2_reg=0.0005, learning_rate=0.005, momentum=0.9):
        self.nlayers = len(layer_sizes) - 1
        self.layers = []

        if hidden_activation=='sigmoid':
            activation_fn = sigmoid
            activation_grad_fn = sigmoid_gradient
        elif hidden_activation=='tanh':
            activation_fn = tanh
            activation_grad_fn = tanh_gradient
        elif hidden_activation=='relu':
            activation_fn = relu
            activation_grad_fn = relu_gradient
        else:
            raise('Unrecognised activation function {}'.format(hidden_activation))

        for l in range(self.nlayers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l+1]
            W = np.random.randn(n_out, n_in) * 2.0 / math.sqrt(n_in + n_out)
            b = np.zeros([n_out, 1])
            layer = {
                'W': W,
                'b': b,
                'fn': activation_fn,
                'fn_grad': activation_grad_fn,
                'dWv': 0,
                'dbv': 0
            }
            self.layers.append(layer)

        self.layers[-1]['fn'] = sigmoid
        self.layers[-1]['fn_grad'] = sigmoid_gradient
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __forward(self, X_batch):
        A = X_batch
        for layer in self.layers:
            Z = np.dot(layer['W'], A) + layer['b']
            A = layer['fn'](Z)
            layer['activation'] = A
        return A

    def __cost(self, Y_batch):
        A = self.layers[-1]['activation']
        return np.mean( Y_batch * np.log(A) + (1.0 - Y_batch) * np.log(1.0-A) )

    def __accuracy(self, Y_batch):
        A = self.layers[-1]['activation']
        Y_pred = (A > 0.5).astype('float')
        return np.mean( (Y_pred == Y_batch) ).astype('float')

    def __fwd_cost(self, X_batch, Y_batch):
        self.__forward(X_batch)
        cost = self.__cost(Y_batch)
        accuracy = self.__accuracy(Y_batch)
        return (cost, accuracy)

    def __backprop_calc_gradients(self, X_batch, Y_batch):
        inv_batch_size = 1.0 / X_batch.shape[1]
        dZ = self.layers[-1]['activation'] - Y_batch
        for l in reversed(range(1,self.nlayers)):
            this_layer = self.layers[l]
            prev_layer = self.layers[l-1]
            A = prev_layer['activation']
            this_layer['dW'] = inv_batch_size * np.dot(dZ, A.T) # ?
            this_layer['db'] = inv_batch_size * np.sum(dZ, axis=1, keepdims=True) # ?
            dZ = prev_layer['fn_grad'](A) * np.dot(this_layer['W'].T, dZ)

        # Layer zero refers X_batch inputs and does not calculate next dZ
        this_layer = self.layers[0]
        this_layer['dW'] = inv_batch_size * np.dot(dZ, X_batch.T) # ?
        this_layer['db'] = inv_batch_size * np.sum(dZ, axis=1, keepdims=True) # ?

    def __apply_weight_change(self):
        for layer in self.layers:
            dW = layer['dW'] + self.l2_reg * layer['W'] # Add L2 regularisation
            layer['dWv'] = layer['dWv'] * self.momentum + dW
            layer['dbv'] = layer['dbv'] * self.momentum + layer['db']
            layer['W'] -= self.learning_rate * layer['dWv']
            layer['b'] -= self.learning_rate * layer['dbv']

    def __train_on_batch(self, X_batch, Y_batch):
        (cost, accuracy) = self.__fwd_cost(X_batch, Y_batch)
        self.__backprop_calc_gradients(X_batch, Y_batch)
        self.__apply_weight_change()
        return (cost, accuracy)

    def fit(self, X_train, y_train, X_cv, y_cv, epochs, batch_size):
        train_accuracies = []
        val_accuracies = []
        train_costs = []
        val_costs = []

        nbatches = int(X_train.shape[1]/batch_size)

        for epoch in range(epochs):
            batch_idx = np.random.permutation(X_train.shape[1])
            train_cost = 0.0
            train_accuracy = 0.0
            for b in range(nbatches):
                this_batch = batch_idx[b*batch_size:(b+1)*batch_size]
                X_batch = X_train[:, this_batch]
                Y_batch = y_train[:, this_batch]
                batch_cost, batch_accuracy = self.__train_on_batch(X_batch, Y_batch)
                train_cost += batch_cost
                train_accuracy += batch_accuracy

            train_cost /= nbatches
            train_accuracy /= nbatches
            val_cost, val_accuracy = self.__fwd_cost(X_cv, y_cv)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_costs.append(train_cost)
            val_costs.append(val_cost)

        return { 'acc': train_accuracies, 'val_acc': val_accuracies,
                 'cost': train_costs, 'val_cost': val_costs }

    def predict_classes(self, X_batch):
        A = self.__forward(X_batch)
        return (A > 0.5).astype('float')

def build_model(layer_sizes, hidden_activation='relu', l2_reg=0.0005, learning_rate=0.005, momentum=0.9):
    return NN(layer_sizes, hidden_activation=hidden_activation, l2_reg=l2_reg,
              learning_rate=learning_rate, momentum=momentum)

def train_model(model, X_train, y_train, X_cv, y_cv, epochs=200, batch_size=10):
    return model.fit(X_train, y_train, X_cv, y_cv, epochs=epochs, batch_size=batch_size)

def test_model(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    accuracy = np.mean( (y_pred == y_test).astype('float') )
    print("Final accuracy {}".format(accuracy * 100))

def plot_history(history):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

def main():
    np.random.seed(1234)
    X_train, y_train, X_cv, y_cv, X_test, y_test = get_data()

    model = build_model( layer_sizes=[X_train.shape[0], 20, 20, 1],
                         hidden_activation='relu',
                         l2_reg=0.0005,
                         learning_rate=0.005,
                         momentum=0.9 )

    history = train_model(model, X_train, y_train, X_cv, y_cv)
    test_model(model, X_test, y_test)
    plot_history(history)

if __name__ == '__main__':
    main()
