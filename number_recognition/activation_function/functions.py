import numpy as np


def Heaviside(z):
    ''' Heaviside function.
        @params:
            z : float values  -- required
        @return:
            1 if z is greater or equal to 0 and 0 in other case'''
    return 1 * (z >= 0)

def d_Heaviside(z):
    return 0


def sgn(z):
    '''Sign function
        @params:
            z : float values -- required
        @return:
            1 if z is greater or equal to 0 and -1 in other case'''
    return 2 * Heaviside(z) - 1

def d_sgn(z):
    return 0


def saturation_function(z):
    ''' Saturation function:
        @params:
            z : float values -- required
        @return:
            1 if z is greater than 1, z if z is lower or equal than 1 and greater
             or equal than -1 and -1 in other case'''
    x = 1 * (z > 1)
    y = -1 * (z < -1)
    w = (abs(z) <= 1) * z
    return x + y + w

def d_saturation_function(z):
    return 1

def logistic(z, s=1):
    ''' Logistic function:
        @params:
            z : float values -- required
            s : weight  -- float values -- required
        @return:
            return the logistic function values '''
    return 1. / (1. + np.exp(-s * z))


def sigmoid(z, s=1):
    ''' Sigmoid function:
        @params:
            z : float values -- required
            s : weight -- float values -- required
        @return:
            return thte sigmoid values'''
    return s * logistic(z, s) * (1 - logistic(z, s))


def tanh(z):
    ''' Hiperbolic tangent function:
        @params:
            z : float values -- required
        @return:
            return the tanh values'''
    return 2 * logistic(z, 2) - 1

def d_tanh(z):
    return 1 - tanh(z) * tanh(z)


def relu(z, smoothed=False):
    ''' REctified Linear Units:
        It's a no lineal activation function that is defined like:
                           [ z  if z >= 0
                    f(z) = |
                           [ 0  if z < 0

        @params:
            z : float values -- required
        @return:
            return the tanh values
        '''
    if smoothed:
        return np.log(1 + np.exp(z))
    else:
        return (z > 0) * z

def d_relu(z, smoothed=False):
    if smoothed:
        return z * np.exp(z) / (1 + relu(z, smoothed))
    else:
        return (z > 0) * 1


act_func = {
            'heaviside': Heaviside,
            'sgn': sgn,
            'saturation_function': saturation_function,
            'sigmoid': logistic,
            'tanh': tanh,
            'relu': relu
}

der_act_func = {
            'heaviside': d_Heaviside,
            'sgn': d_sgn,
            'saturation_function': d_saturation_function,
            'sigmoid': sigmoid,
            'tanh': d_tanh,
            'relu': d_relu
}