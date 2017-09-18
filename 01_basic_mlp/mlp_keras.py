import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

def build_model(layer_sizes, hidden_activation='relu', l2_reg=0.0005):
    model = Sequential()
    # First layer needs input dimension in Keras
    model.add(Dense(layer_sizes[1], input_dim=layer_sizes[0], kernel_regularizer=l2(l2_reg)))

    for size in layer_sizes[2:]:
        model.add(Activation(hidden_activation))
        model.add(Dense(size, kernel_regularizer=l2(l2_reg)))

    model.add(Activation('sigmoid'))
    return model

def set_objective(model, learning_rate=0.005, momentum=0.9):
    sgd = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

def train_model(model, X_train, y_train, X_cv, y_cv, epochs=200, batch_size=10):
    history = model.fit(X_train, y_train, validation_data=(X_cv, y_cv), epochs=epochs, batch_size=batch_size, verbose=0)
    return history.history

def test_model(model, X_test, y_test):
    y_pred = model.predict_classes(X_test, verbose=0)
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
    X, y = sklearn.datasets.make_moons(600, noise=0.30)
    y = y.reshape([600,1])
    X_train = X[:400]; y_train = y[:400]
    X_cv = X[400:500]; y_cv = y[400:500]
    X_test = X[500:]; y_test = y[500:]

    model = build_model([X.shape[1], 20, 20, 1])
    set_objective(model)
    history = train_model(model, X_train, y_train, X_cv, y_cv)
    test_model(model, X_test, y_test)
    plot_history(history)

if __name__ == '__main__':
    main()
