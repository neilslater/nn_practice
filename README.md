# nn_practice

Collection of Python scripts using different NN frameworks. They are based on work from various
online courses and tutorials, and intended to be used as a kind of kata, as opposed to a re-usable
library, although most would work as a starting point for handling specific problems.

## 01_basic_mlp

Feed-forward networks solving basic problems, implemented using different libraries. This
includes code that just uses Numpy and basic matrix operations. The minimal spec is arbitrarily chosen,
and covers:

 * NN model must be a Python class, taking parameters controlling number of layers, layer size, activation of hidden layers

 * Binary classification only, optimised against cross-entropy loss function

 * L2 regularisation

 * Optimisation using SGD with Nesterov momentum, applied in mini-batches. Parameters controlling
 learning rate, momentum and mini-batch size

 * Basic train/test split in order to assess model.

 * Using "moons" generated data from `sklearn`.

 * Plot of learning curve
