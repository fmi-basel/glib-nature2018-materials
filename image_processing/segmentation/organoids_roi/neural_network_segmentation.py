import abc
import os
import logging

import six
from tqdm import tqdm
from skimage.external.tifffile import imsave

import numpy as np

from dlutils.preprocessing.normalization import standardize
from dlutils.prediction import predict_complete
from dlutils.models import load_model

from image_processing.reader.stackreader import StackGenerator


# TODO consider moving these to dlutils.
def get_idx_range(idx, dx, max_idx):
    '''indices to get dx slices above and below given center.

    '''
    return [max(0, min(x, max_idx - 1)) for x in range(idx - dx, idx + dx + 1)]


def get_multislice(stack, idx, dx=1):
    '''get multislice substack from stack centered at idx with dx slices
    above and below.

    NOTE stack is transformed from ZYX to YXZ for compatibility with
    downstream (tensorflow-based) models.

    '''
    img = stack[get_idx_range(idx, dx, len(stack))]
    img = np.moveaxis(img, 0, -1)
    return img


def imsave_prediction(path, probs):
    '''save prediction rescaled from [0, 1] to [0, 255].

    '''
    return imsave(path, (probs * 255).astype(np.uint8))


def is_multislice_model(model):
    '''determine if the model expects multislice inputs.

    '''
    input_shape = model.input_shape
    if isinstance(input_shape, tuple):
        channel_dim = input_shape[-1]
    elif isinstance(input_shape, list):
        channel_dim = input_shape[0][-1]
    else:
        raise ValueError(
            'Could not determine if model is multislice. Input shape: {}'.
            format(input_shape))
    return channel_dim >= 3


@six.add_metaclass(abc.ABCMeta)
class BaseNeuralNetworkRoiSegmentation():
    '''Base for neural network segmentation of 3D ROI stacks.

    '''

    @property
    @abc.abstractmethod
    def model_path(self):
        '''specifies path to trained model.

        '''
        pass

    @property
    @abc.abstractmethod
    def input_channel_pattern(self):
        '''filename pattern for required channel.

        '''
        pass

    @abc.abstractmethod
    def processor_fn(self, stack):
        '''applies model to stack.

        '''
        pass

    @abc.abstractmethod
    def save_fn(self, stack, pred_stack, out_dir):
        '''saves predictions for a given stack.

        '''
        pass

    def __init__(self):
        '''
        '''
        self.load_model()

    def load_model(self):
        '''
        '''
        logging.getLogger(__name__).info('Loading model from %s ...',
                                         self.model_path)
        self.model = load_model(self.model_path)
        self.is_multislice = is_multislice_model(self.model)
        logging.getLogger(__name__).info('Loading successful.')

    def segment(self, stack_dir, out_dir):
        '''applies the segmentation model to the given stack_dir and
        saves output to out_dir.

        :param stack_dir: path to stack
        :param out_dir: path to save outputs

        '''
        # load stacks
        logging.getLogger(__name__).info('Processing folder: %s', stack_dir)

        if os.path.abspath(stack_dir) == os.path.abspath(out_dir):
            raise ValueError('Input and output directory cannot be the same!')

        for stack in self.generator_fn(stack_dir):
            pred_stack = self.processor_fn(stack)
            self.save_fn(stack=stack, pred_stack=pred_stack, out_dir=out_dir)

    def generator_fn(self, stack_dir):
        '''reads input stack and normalizes images.

        '''
        generator = StackGenerator(
            img_dir=stack_dir,
            segm_dir=None,
            pattern=[self.input_channel_pattern, '*mask.tif'])

        for stack in tqdm(generator, leave=True, desc='Processing well', ncols=80):
            stack['image_stack'] = standardize(stack['image_stack'], 50)
            yield stack


class NeuralNetworkSegmentationSP(BaseNeuralNetworkRoiSegmentation):
    '''single plane organoid segmentation.
    '''

    # path to trained network.
    model_base_path = os.path.join(
        os.path.dirname(__file__), 'networks', 'single_plane')
    model_path = os.path.join(model_base_path, 'v_1', 'model_best.h5')

    # filename pattern for required channel.
    input_channel_pattern = '*C04.tif'

    def processor_fn(self, stack):
        '''apply model to stack.

        '''
        pred_stack = []
        for idx in range(len(stack['image_paths'])):

            if self.is_multislice:
                img = get_multislice(stack['image_stack'], idx)
            else:
                img = stack['image_stack'][idx]
            pred = predict_complete(self.model, img, batch_size=1)

            # apply mask.
            pred_stack.append({
                'cell_pred':
                pred['cell_pred_c1x1'].squeeze() * stack['mask']
            })

        return pred_stack

    def save_fn(self, stack, pred_stack, out_dir):
        '''saves all predicted channels in individual subfolders.

        '''
        assert len(stack['image_paths']) == len(pred_stack)
        for path, pred in zip(stack['image_paths'], pred_stack):
            imsave_prediction(
                self.get_outpath(out_dir, path), pred['cell_pred'])

    @staticmethod
    def get_outpath(dirname, path):
        '''generate path to save predictions to.

        '''
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            logging.getLogger(__name__).debug('Creating output folder at %s',
                                              dirname)
        return os.path.join(dirname, os.path.basename(path))


class NeuralNetworkSegmentationSC(BaseNeuralNetworkRoiSegmentation):
    '''single cell nucleus segmentation.
    '''

    # path to trained network.
    model_base_path = os.path.join(
        os.path.dirname(__file__), 'networks', 'single_cell')
    model_path = os.path.join(model_base_path, 'v_1', 'model_best.h5')

    # filename pattern for required channel.
    input_channel_pattern = '*C01.tif'

    def processor_fn(self, stack):
        '''
        '''
        pred_stack = []
        for idx in range(len(stack['image_paths'])):

            if self.is_multislice:
                img = get_multislice(stack['image_stack'], idx)
            else:
                img = stack['image_stack'][idx]
            pred = predict_complete(self.model, img, batch_size=1)
            pred_stack.append(pred)

        return pred_stack

    def save_fn(self, stack, pred_stack, out_dir):
        '''saves all predicted channels in individual subfolders.

        '''
        assert len(stack['image_paths']) == len(pred_stack)
        for path, pred in zip(stack['image_paths'], pred_stack):
            for key, val in pred.items():
                imsave_prediction(self.get_outpath(out_dir, key, path), val)

    @staticmethod
    def get_outpath(out_dir, key, path):
        '''
        '''
        dirname = os.path.join(out_dir, key)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            logging.getLogger(__name__).debug('Creating output folder at %s',
                                              dirname)
        return os.path.join(dirname, os.path.basename(path))
