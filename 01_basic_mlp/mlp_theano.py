import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import math
import theano
import theano.tensor as T

def get_data():
    X, y = sklearn.datasets.make_moons(600, noise=0.30)
    y = y.reshape([600,1])
    X_train = X[:400]; y_train = y[:400]
    X_cv = X[400:500]; y_cv = y[400:500]
    X_test = X[500:]; y_test = y[500:]

    return (X_train, y_train, X_cv, y_cv, X_test, y_test)

class NN():
    def __init__(self, layer_sizes, hidden_activation='relu', l2_reg=0.0005, learning_rate=0.005, momentum=0.9):
        self.nlayers = len(layer_sizes) - 1
        self.layers = []

        X = T.matrix("X")
        Y = T.matrix("Y")

        if hidden_activation=='sigmoid':
            activation_fn = T.nnet.sigmoid
        elif hidden_activation=='tanh':
            activation_fn = T.tanh
        elif hidden_activation=='relu':
            activation_fn = T.nnet.relu
        else:
            raise('Unrecognised activation function {}'.format(hidden_activation))

        prev_A = X

        for l in range(self.nlayers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l+1]

            W_init = np.random.randn(n_in, n_out) * math.sqrt(2.0/(n_in + n_out))
            b_init = np.zeros([n_out])
            dWv_init = np.zeros([n_in, n_out])
            dbv_init = np.zeros([n_out])

            W = theano.shared(W_init, name='W{}'.format(l+1))
            b = theano.shared(b_init, name='b{}'.format(l+1))
            dWv = theano.shared(dWv_init, name='dWv{}'.format(l+1))
            dbv = theano.shared(dbv_init, name='dbv{}'.format(l+1))

            Z = prev_A.dot(W) + b
            if l + 1 <  self.nlayers:
                A = activation_fn(Z)
            else:
                A = T.nnet.sigmoid(Z)

            layer = {
                'W': W,
                'b': b,
                'dWv': dWv,
                'dbv': dbv,
                'Z': Z,
                'A': A
            }
            self.layers.append(layer)

            prev_A = A

        logits = Z
        Y_pred = A > 0.5
        accuracy = T.eq(Y_pred, Y).mean()
        cost = T.nnet.nnet.binary_crossentropy(A, Y).mean()

        nesterov_step_a_updates = []
        nesterov_step_b_updates = []

        for layer in self.layers:
            nesterov_step_a_updates.append( (layer['W'], layer['W'] - learning_rate * layer['dWv']) )
            nesterov_step_a_updates.append( (layer['b'], layer['b'] - learning_rate * layer['dbv']) )

            dW = T.grad(cost, layer['W']) + l2_reg * layer['W']
            db = T.grad(cost, layer['b'])

            nesterov_step_b_updates.append( (layer['W'], layer['W'] - learning_rate * dW) )
            nesterov_step_b_updates.append( (layer['b'], layer['b'] - learning_rate * db) )
            nesterov_step_b_updates.append( (layer['dWv'], layer['dWv'] * momentum + dW) )
            nesterov_step_b_updates.append( (layer['dbv'], layer['dbv'] * momentum + db) )

        self.predict_classes = theano.function(inputs=[X], outputs=[A])
        self.test_metrics = theano.function(inputs=[X, Y], outputs=[cost, accuracy])

        self.__nesterov_step_a = theano.function(
            inputs=[],
            updates=nesterov_step_a_updates
        )

        self.__nesterov_step_b = theano.function(
            inputs=[X, Y],
            updates=nesterov_step_b_updates,
            outputs=[cost, accuracy]
        )

    def __train_on_batch(self, X_batch, Y_batch):
        self.__nesterov_step_a()
        return self.__nesterov_step_b(X_batch, Y_batch)

    def fit(self, X_train, y_train, X_cv, y_cv, epochs, batch_size):
        train_accuracies = []
        val_accuracies = []
        train_costs = []
        val_costs = []

        nbatches = int(X_train.shape[0]/batch_size)

        for epoch in range(epochs):
            batch_idx = np.random.permutation(X_train.shape[0])
            train_cost = 0.0
            train_accuracy = 0.0
            for b in range(nbatches):
                this_batch = batch_idx[b*batch_size:(b+1)*batch_size]
                X_batch = X_train[this_batch, :]
                Y_batch = y_train[this_batch, :]

                batch_cost, batch_accuracy = self.__train_on_batch(X_batch, Y_batch)
                train_cost += batch_cost
                train_accuracy += batch_accuracy

            train_cost /= nbatches
            train_accuracy /= nbatches
            val_cost, val_accuracy = self.test_metrics(X_cv, y_cv)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_costs.append(train_cost)
            val_costs.append(val_cost)

        return { 'acc': train_accuracies, 'val_acc': val_accuracies,
                 'cost': train_costs, 'val_cost': val_costs }


def build_model(layer_sizes, hidden_activation='relu', l2_reg=0.0005, learning_rate=0.005, momentum=0.9):
    return NN(layer_sizes, hidden_activation=hidden_activation, l2_reg=l2_reg,
              learning_rate=learning_rate, momentum=momentum)

def train_model(model, X_train, y_train, X_cv, y_cv, epochs=200, batch_size=10):
    return model.fit(X_train, y_train, X_cv, y_cv, epochs=epochs, batch_size=batch_size)

def test_model(model, X_test, y_test):
    cost, accuracy = model.test_metrics(X_test, y_test)
    print("Final accuracy {:0.1f}".format(accuracy * 100))

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

    model = build_model( layer_sizes=[X_train.shape[1], 20, 20, 1],
                         hidden_activation='relu',
                         l2_reg=0.0005,
                         learning_rate=0.005,
                         momentum=0.9 )

    history = train_model(model, X_train, y_train, X_cv, y_cv)
    test_model(model, X_test, y_test)
    plot_history(history)

if __name__ == '__main__':
    main()
