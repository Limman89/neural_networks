from activation_function import *


activation_function = act_func


class Perceptron(object):
    """Perceptron classifier

    @params:
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : integer
            Passes over the training dataset
        random_state : integer
            Random number generator seed for random weigth initialization
        func : string
            Activation function
        args : args
            Activation function extra arguments
    @attributes:
        w_ : 1d-array
            Weights after fitting
        errors_ : list
            Number of misclassifications (updates) in each epoch

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1, func='sgn'):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.act_func = func

    def fit(self, X, y):
        """Fit training data.

        @params:
            X : {array-like}, shape = [n_samples, n_featurs]
                Training vectors, where n_samples is the number of samples and n_features is the number of features.
            y : array-like, shape = [n_samples]
                Target values.
        @outputs:
            self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        #return np.where(self.net_input(X) >= 0.0, 1, -1)
        return activation_function[self.act_func](self.net_input(X))



class Sigmoidal_Perceptron(object):
    """Perceptron classifier

    @params:
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : integer
            Passes over the training dataset
        random_state : integer
            Random number generator seed for random weigth initialization
        func : string
            Activation function
        args : args
            Activation function extra arguments
    @attributes:
        w_ : 1d-array
            Weights after fitting
        errors_ : list
            Number of misclassifications (updates) in each epoch

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1, func='sigmoidal', s=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.act_func = func
        self.s = s

    def fit(self, X, y):
        """Fit training data.

        @params:
            X : {array-like}, shape = [n_samples, n_featurs]
                Training vectors, where n_samples is the number of samples and n_features is the number of features.
            y : array-like, shape = [n_samples]
                Target values.
        @outputs:
            self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) * (1 - self.predict(xi) ** 2)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return activation_function[self.act_func](self.net_input(X), self.s)

