import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import math
import torch
from torch import nn

def get_data():
    X, y = sklearn.datasets.make_moons(600, noise=0.30)
    y = y.reshape([600,1])
    X_train = X[:400]; y_train = y[:400]
    X_cv = X[400:500]; y_cv = y[400:500]
    X_test = X[500:]; y_test = y[500:]
    return (X_train, y_train, X_cv, y_cv, X_test, y_test)

class NN():
    def __init__(self, layer_sizes, hidden_activation='relu', l2_reg=0.0005, learning_rate=0.005, momentum=0.9):
        layers = []
        nlayers = len(layer_sizes) - 1

        for l in range(nlayers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l+1]
            layers.append(nn.Linear(n_in, n_out))
            activation = hidden_activation
            if l == nlayers - 1:
                activation = 'sigmoid'

            if activation=='sigmoid':
                layers.append(nn.Sigmoid())
            elif activation=='tanh':
                layers.append(nn.Tanh())
            elif activation=='relu':
                layers.append(nn.ReLU())
            elif activation=='linear':
                # Do nothing
                layers
            else:
                raise('Unrecognised activation function {}'.format(activation))

        self.model = nn.Sequential(*layers)
        self.loss = nn.BCELoss()   # TODO: Use BCEWithLogitsLoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                         momentum=momentum, weight_decay=l2_reg, nesterov=True)

    def __train_on_batch(self, X_batch, Y_batch):
        xt = torch.Tensor(X_batch)
        yt = torch.Tensor(Y_batch)
        out = self.model(xt)
        loss_val = self.loss(out, yt) # Change when using BCEWithLogitsLoss()
        loss_val.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()
        batch_accuracy = (yt.byte() == (out > 0.5)).float().mean().item()
        return (loss_val.item(), batch_accuracy)

    def __fwd_cost(self, X_batch, Y_batch):
        xt = torch.Tensor(X_batch)
        yt = torch.Tensor(Y_batch)
        out = self.model(xt)
        loss_val = self.loss(out, yt) # Change when using BCEWithLogitsLoss()
        batch_accuracy = (yt.byte() == (out > 0.5)).float().mean()
        return (loss_val.item(), batch_accuracy)

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

            val_cost, val_accuracy = self.__fwd_cost(X_cv, y_cv)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_costs.append(train_cost)
            val_costs.append(val_cost)

        return { 'acc': train_accuracies, 'val_acc': val_accuracies,
                 'cost': train_costs, 'val_cost': val_costs }

    def predict_classes(self, X_batch):
        A = self.model(torch.Tensor(X_batch))
        return np.array(A > 0.5).astype('float') # Probably don't need to convert

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
