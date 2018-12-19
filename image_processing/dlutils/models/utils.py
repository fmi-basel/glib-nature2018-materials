from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_crop_shape(x_shape, y_shape):
    '''determine crop delta for a concatenation.

    NOTE Assumes that y is larger than x.
    '''
    assert len(x_shape) == len(y_shape)
    assert len(x_shape) >= 2
    shape = []

    for xx, yy in zip(x_shape, y_shape):
        delta = yy - xx
        if delta < 0:
            delta = 0
        if delta % 2 == 1:
            shape.append((int(delta / 2), int(delta / 2) + 1))
        else:
            shape.append((int(delta / 2), int(delta / 2)))
    return shape


def get_batch_size(model):
    '''
    '''
    return model.input_shape[0]


def get_patch_size(model):
    '''
    '''
    return model.input_shape[1:-1]


def get_input_channels(model):
    '''
    '''
    return model.input_shape[-1]
