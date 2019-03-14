'''simple in-plane classification with a logistic regression.
'''

import numpy as np


def calc_max_ratio(stack):
    '''ratio of inplane max-intensity to max-intensity of stack.

    '''
    return np.max(stack, axis=(1, 2)) / stack.max()


def logistic_reg(val, loc=-2.754005, scale=8.635463):
    '''parameters from 'fit out of focus model'
    
    '''
    val = val * scale + loc
    return 1. / (1. + np.exp(-val))


def predict_infocus(stack):
    '''predict probability for each XY slice of the stack to be in-focus.

    Parameters
    ----------
    stack : array_like, shape=(Z, X, Y )
        image stack.
    
    Returns
    -------
    probability : array_like, shape=(Z, )
        predicted in-focus probability.

    '''
    # Parameters from fit in 'fit out of focus model.ipynb'
    loc = -2.754005
    scale = 8.635463

    return logistic_reg(calc_max_ratio(stack), loc=loc, scale=scale)
