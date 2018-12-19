from keras.models import load_model as keras_load_model
from keras.models import model_from_yaml, model_from_json

from image_processing.dlutils.layers.grouped_conv import GroupedConv2D
from image_processing.dlutils.layers.padding import DynamicPaddingLayer
from image_processing.dlutils.layers.padding import DynamicTrimmingLayer

import os
import logging

CUSTOM_LAYERS = {
    'GroupedConv2D': GroupedConv2D,
    'DynamicPaddingLayer': DynamicPaddingLayer,
    'DynamicTrimmingLayer': DynamicTrimmingLayer
}


def load_model(file_path, *args, **kwargs):
    '''load a previously saved model.

    '''
    custom_objects = kwargs.pop('custom_objects', dict())
    custom_objects.update(CUSTOM_LAYERS)

    logger = logging.getLogger(__name__)

    for loader in [keras_load_model, load_from_yaml_with_weights]:
        try:
            return loader(
                file_path, *args, custom_objects=custom_objects, **kwargs)
        except ValueError as err:
            logger.debug('{} did not succeed: {}'.format(
                loader.__name__, str(err)))
    raise ValueError('Could not load model from {}'.format(file_path))


def find_architecture_and_weight_paths(path):
    '''
    '''
    basename, ext = os.path.splitext(path)
    if ext in ['.yaml', '.json']:
        file_path = path
        weight_path = os.path.join(os.path.dirname(path), 'model_latest.h5')
    elif ext in ['.h5', '.hdf5']:
        file_path = os.path.join(
            os.path.dirname(path), 'model_architecture.yaml')
        weight_path = path
    return file_path, weight_path


def load_from_yaml_with_weights(file_path,
                                weight_path=None,
                                custom_objects=None,
                                **kwargs):
    '''
    '''
    if weight_path is None:
        file_path, weight_path = find_architecture_and_weight_paths(file_path)
    with open(file_path, 'r') as fin:
        model = model_from_yaml(fin.read(), custom_objects=custom_objects)
    model.load_weights(weight_path)
    return model


def load_from_json_with_weights(file_path,
                                weight_path=None,
                                custom_objects=None,
                                **kwargs):
    '''
    '''
    if weight_path is None:
        file_path, weight_path = find_architecture_and_weight_paths(file_path)
    with open(file_path, 'r') as fin:
        model = model_from_json(fin.read(), custom_objects=custom_objects)
    model.load_weights(weight_path)
    return model
