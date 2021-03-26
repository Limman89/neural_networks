from activation_function.functions import *


class Activation:
    def __init__(self, function):
        self.function = function

    def eval(self, z):
        """ Evaluate the activation function."""
        return act_func[self.function](z)

    def diff(self, z):
        """ Evaluate the derivative of the activation function."""
        return der_act_func[self.function](z)