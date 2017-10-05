import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import math
import tensorflow as tf

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

        self.X = tf.placeholder(tf.float32, shape=[None, layer_sizes[0]], name="X")
        self.Y = tf.placeholder(tf.float32, shape=[None, layer_sizes[-1]], name="Y")

        if hidden_activation=='sigmoid':
            activation_fn = tf.nn.sigmoid
        elif hidden_activation=='tanh':
            activation_fn = tf.nn.tanh
        elif hidden_activation=='relu':
            activation_fn = tf.nn.relu
        else:
            raise('Unrecognised activation function {}'.format(hidden_activation))

        prev_A = self.X

        self.wl2 = 0

        for l in range(self.nlayers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l+1]
            scale = math.sqrt(2/(n_in + n_out))

            W = tf.Variable(tf.random_normal([n_in, n_out]) * scale, name='W{}'.format(l+1))
            b = tf.Variable(tf.zeros([n_out]), name='b{}'.format(l+1))
            Z = tf.add(tf.matmul(prev_A, W), b, name='Z{}'.format(l+1))
            if l + 1 <  self.nlayers:
                A = activation_fn(Z, name='A{}'.format(l+1))
            else:
                A = tf.nn.sigmoid(Z, name='Y_prob')

            self.wl2 = self.wl2 + tf.nn.l2_loss(W)
            prev_A = A

        self.logits = Z
        self.Y_pred = tf.cast(tf.greater(A, 0.5, name='Y_pred'), 'float')

        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        self.cost_plus_wl2 = self.cost + l2_reg * self.wl2

        self.correct = tf.equal(self.Y_pred, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
        self.train = self.optimizer.minimize(self.cost_plus_wl2)

    def __train_on_batch(self, X_batch, Y_batch):
        _, cost, accuracy = self.sess.run([self.train, self.cost, self.accuracy],
                                 feed_dict={self.X: X_batch, self.Y: Y_batch})
        return (cost, accuracy)

    def fit(self, X_train, y_train, X_cv, y_cv, epochs, batch_size):
        train_accuracies = []
        val_accuracies = []
        train_costs = []
        val_costs = []

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

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
            val_cost, val_accuracy = self.sess.run([ self.cost, self.accuracy],
                                 feed_dict={self.X: X_cv, self.Y: y_cv})

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_costs.append(train_cost)
            val_costs.append(val_cost)

        return { 'acc': train_accuracies, 'val_acc': val_accuracies,
                 'cost': train_costs, 'val_cost': val_costs }

    def predict_classes(self, X_batch):
        Y_pred = self.sess.run(self.Y_pred, feed_dict={self.X: X_batch})
        return Y_pred

    def test_accuracy(self, X_batch, Y_batch):
        return self.sess.run(self.accuracy, feed_dict={self.X: X_batch, self.Y: Y_batch})

def build_model(layer_sizes, hidden_activation='relu', l2_reg=0.0005, learning_rate=0.005, momentum=0.9):
    return NN(layer_sizes, hidden_activation=hidden_activation, l2_reg=l2_reg,
              learning_rate=learning_rate, momentum=momentum)

def train_model(model, X_train, y_train, X_cv, y_cv, epochs=200, batch_size=10):
    return model.fit(X_train, y_train, X_cv, y_cv, epochs=epochs, batch_size=batch_size)

def test_model(model, X_test, y_test):
    accuracy = model.test_accuracy(X_test, y_test)
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
