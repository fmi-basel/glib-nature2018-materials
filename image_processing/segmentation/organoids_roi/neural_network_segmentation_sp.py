import os
import logging

import numpy as np
from tqdm import tqdm
from skimage.external.tifffile import imsave as tiffsave

from dlutils.models import load_model
from dlutils.preprocessing.normalization import standardize

from image_processing.reader.stackreader import StackGenerator

# logger format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S')

# disable tensorflow clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NeuralNetworkSegmentationSP():

    def imsave(self, path, img, rescale):
        '''
        '''
        if rescale:
            img = (img * 255).astype(np.uint8)
        tiffsave(path, img)

    def get_outpath(self, dirname, path):
        '''generate path to save predictions to.

        '''
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return os.path.join(dirname, os.path.basename(path))

    def save_oof(self, outdir, data_dict, delimiter=','):
        '''write a set of pairs to a .csv file.

        '''
        with open(os.path.join(outdir, 'oof_predictions.csv'), 'w') as fout:
            for key, val in data_dict.items():
                fout.write(
                    delimiter.join([str(key), '{:1.2f}'.format(val)]) + '\n')

    def predict_flat(self, model, image, mask):
        '''
        '''
        pred = dict(
            zip(
                model.output_names,
                model.predict({
                    'input': image[None, ...],
                    'mask': mask[None, ...]
                })))
        for key, val in pred.items():
            if isinstance(val, np.ndarray):
                pred[key] = val.squeeze(axis=0)

        return pred

    def predict(self, model, image, mask, batch_size=10, border=20, patch_size=None):
        '''apply trained multi-task model to image.

        NOTE Expects the following model outputs:

        - cell_pred : array_like
        - oof_pred : scalar

        '''
        from dlutils.prediction.stitching import StitchingGenerator

        assert image.shape[:2] == mask.shape[:2]

        if patch_size is None:
            patch_size = model.input_shape[0][1:-1]

        # add "flat" channel if necessary
        if image.ndim == 2:
            image = image[..., None]
        if mask.ndim == 2:
            mask = mask[..., None]

        if all(y is None or x == y for x, y in zip(image.shape, patch_size)):
            return self.predict_flat(model, image, mask)

        # check if the patch_size fits within image.shape
        diff_shape = [max(x - y, 0) for x, y in zip(patch_size, image.shape)]

        if border > 0 or any(diff_shape > 0):
            pad_width = [(
                border + dx // 2,
                border + dx // 2 + dx % 2,
            ) for idx, dx in enumerate(diff_shape)] + [
                            (0, 0),
                        ]
            image = np.pad(image, pad_width=pad_width, mode='symmetric')
            mask = np.pad(mask, pad_width=pad_width, mode='symmetric')

        img_generator = StitchingGenerator(
            image, patch_size=patch_size, batch_size=batch_size, border=border)
        mask_generator = StitchingGenerator(
            mask, patch_size=patch_size, batch_size=batch_size, border=border)

        responses = dict(
            zip(model.output_names, [np.zeros(image.shape[:-1] + (1,)), []]))

        for img_batch, mask_batch in ((img_generator[idx],
                                       mask_generator[idx]['input'])
                                      for idx in range(len(img_generator))):

            img_batch, coord_batch = img_batch['input'], img_batch['coord']

            pred_batch = model.predict_on_batch({
                'input': img_batch,
                'mask': mask_batch
            })
            pred_batch = dict(zip(model.output_names, pred_batch))

            # re-assemble segmentation
            for idx, coord in enumerate(coord_batch):
                slices = tuple([
                    slice(x + border, x + dx - border)
                    for x, dx in zip(coord, patch_size)
                ])

                key = 'cell_pred'
                pred = pred_batch[key]

                border_slices = tuple(
                    [slice(border, -border) for _ in range(pred[idx].ndim - 1)])

                # TODO implement smooth stitching.
                responses[key][slices] = pred[idx][border_slices]

            responses['oof_pred'].extend(pred_batch['oof_pred'])

        # mean prediction for oof_pred
        responses['oof_pred'] = np.mean(responses['oof_pred'])

        # crop if we padded earlier
        if border > 0 or any(np.asarray(diff_shape) > 0):
            slices = tuple([
                slice(border + dx // 2, -(border + dx // 2 + dx % 2))
                for dx in diff_shape
            ])

            for key, val in responses.items():
                if isinstance(val, np.ndarray):
                    responses[key] = val[slices]

        return responses


    def segment(self, stack_dir):
        '''
        '''
        model_base_path = os.path.join(os.path.dirname(__file__), 'networks')
        model_path = os.path.join(model_base_path, 'v_1', 'model_best.h5')
        logging.info('Loading model from {} ...'.format(model_path))

        model = load_model(model_path)

        logging.info('successful!')
        is_multislice = model.input_shape[0][-1] >= 3
        if is_multislice:
            idx_dx = model.input_shape[0][-1] // 2

        # load stacks
        logging.getLogger(__name__).info('Processing folder: %s', stack_dir)

        outdir = os.path.join(os.path.dirname(stack_dir), 'roi_pred')
        logging.getLogger(__name__).info('Output is being written to: %s', outdir)

        if os.path.abspath(stack_dir) == os.path.abspath(outdir):
            raise ValueError('Input and output directory cannot be the same!')

        for stack in tqdm(
                StackGenerator(img_dir=stack_dir, segm_dir=None), ncols=80):

            images = standardize(stack['image_stack'], min_scale=50)

            for idx, path in tqdm(
                    enumerate(stack['image_paths']),
                    desc='Processing stack',
                    total=len(images),
                    ncols=80,
                    leave=False):

                if is_multislice:
                    indices = [
                        max(min(x, images.shape[0] - 1), 0)
                        for x in range(idx - idx_dx, idx + idx_dx + 1)
                    ]
                    image = images[indices]
                    image = np.moveaxis(image, 0, -1)
                else:
                    image = images[idx]

                prediction = self.predict(
                    model, image, stack['mask'], border=50, batch_size=1)

                out_path = self.get_outpath(outdir, path)
                self.imsave(out_path, prediction['cell_pred'], rescale=True)
