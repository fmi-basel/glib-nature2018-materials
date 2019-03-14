from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

from image_processing.dlutils.models.utils import get_batch_size
from image_processing.dlutils.models.utils import get_patch_size
from image_processing.dlutils.models.utils import get_input_channels

from keras.utils import Sequence

import numpy as np


class StitchingGenerator(Sequence):
    def __init__(self, image, batch_size, patch_size, border):
        '''
        '''
        self.image = image
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.border = border if batch_size is not None else 1
        self.calc_corners()

    def __len__(self):
        '''determine the number of batches needed.

        '''
        return int(np.ceil(len(self.corners) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''generates the ith-batch of patches.

        '''
        # TODO integrate pre-selection of patches to only process those
        # patches that contain foreground
        if idx > len(self):
            raise IndexError('idx {} out of range {}'.format(idx, len(self)))
        batch_end = min((idx + 1) * self.batch_size, len(self.corners))
        coord_batch, img_batch = list(
            zip(*[((i, j), self.image[i:i + self.patch_size[0], j:j +
                                      self.patch_size[1], ...])
                  for i, j in self.corners[idx * self.batch_size:batch_end]]))

        return dict(input=np.asarray(img_batch), coord=np.asarray(coord_batch))

    def calc_corners(self):
        '''
        '''
        # corners of patches.
        step_size = np.asarray(self.patch_size) - 2 * self.border
        x = list(
            range(0, self.image.shape[0] - self.patch_size[0], step_size[0]))
        y = list(
            range(0, self.image.shape[1] - self.patch_size[1], step_size[1]))
        x.append(self.image.shape[0] - self.patch_size[0])
        y.append(self.image.shape[1] - self.patch_size[1])
        self.corners = [(i, j) for i in x for j in y]




def predict_complete(model, image, batch_size=None, patch_size=None,
                     border=10):
    '''apply model to entire image.

    '''
    if batch_size is None:
        batch_size = get_batch_size(model)
    if patch_size is None:
        patch_size = get_patch_size(model)
    n_channels = get_input_channels(model)

    if batch_size is None:
        raise RuntimeError('Couldnt determine batch_size!')
    if patch_size is None:
        raise RuntimeError('Couldnt determine patch_size!')

    # add "flat" channel if necessary
    if n_channels == 1 and image.shape[-1] != 1:
        image = image[..., None]

    # predict complete image at once.
    if all(y is None or x == y for (x, y) in zip(image.shape, patch_size)):
        if len(model.output_names) == 1:
            return {model.output_names[0]: model.predict(image[None, ...])}

        pred = dict(zip(model.output_names, model.predict(image[None, ...])))
        for key, val in pred.items():
            pred[key] = val.squeeze(axis=0)
        return pred

    # check if the patch_size fits within image.shape
    diff_shape = [max(x - y, 0) for x, y in zip(patch_size, image.shape)]

    if border > 0 or any(val > 0 for val in diff_shape):
        pad_width = [(
            border + dx // 2,
            border + dx // 2 + dx % 2,
        ) for idx, dx in enumerate(diff_shape)] + [
            (0, 0),
        ]
        image = np.pad(image, pad_width=pad_width, mode='symmetric')

    # predict on each patch.
    # TODO consider stitching and prediction concurrently.
    # TODO allow for prediction-time-augmentation
    generator = StitchingGenerator(
        image, patch_size=patch_size, batch_size=batch_size, border=border)

    if len(model.output_names) > 1:
        responses = dict(
            (name, np.zeros(image.shape[:-1] + (out_shape[-1], )))
            for name, out_shape in zip(model.output_names, model.output_shape))
    else:
        responses = {
            model.output_names[0]:
            np.zeros(image.shape[:-1] + (model.output_shape[-1], ))
        }

    for img_batch, coord_batch in (
        (batch['input'], batch['coord'])
            for batch in (generator[idx] for idx in range(len(generator)))):

        # predict
        pred_batch = model.predict_on_batch(img_batch)

        # if we have only one output, then the return is not a list.
        if len(model.output_names) == 1:
            pred_batch = pred_batch[None, ...]

        # re-assemble
        for idx, coord in enumerate(coord_batch):
            slices = tuple([
                slice(x + border, x + dx - border)
                for x, dx in zip(coord, patch_size)
            ])

            for key, pred in zip(model.output_names, pred_batch):

                border_slices = tuple([
                    slice(border, -border) for _ in range(pred[idx].ndim - 1)
                ])

                # TODO implement smooth stitching.
                responses[key][slices] = pred[idx][border_slices]

    if border > 0 or any(np.asarray(diff_shape) > 0):

        slices = tuple([
            slice(border + dx // 2, -(border + dx // 2 + dx % 2))
            for dx in diff_shape
        ])

        for key, val in responses.items():
            responses[key] = val[slices]

    return responses
