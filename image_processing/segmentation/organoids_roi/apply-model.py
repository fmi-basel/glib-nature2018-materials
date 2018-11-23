'''Segment single-plane organoids

'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import logging

import numpy as np
from tqdm import tqdm
from skimage.external.tifffile import imsave as tiffsave

from dlutils.models import load_model
from dlutils.preprocessing.normalization import standardize

from features.stackreader import StackGenerator

# logger format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S')

# disable tensorflow clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse():
    '''parse command line arguments.
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='folder with stack', required=True)
    parser.add_argument('--output', help='output directory', required=True)
    parser.add_argument('--model', help='model to load', required=True)

    args = parser.parse_args()
    logging.getLogger(__name__).debug('Parsed arguments:')
    logging.getLogger(__name__).debug('  model=%s', args.model)
    logging.getLogger(__name__).debug('  inputs=%s', args.input)
    logging.getLogger(__name__).debug('  output=%s', args.output)
    return args


def get_outpath(dirname, path):
    '''generate path to save predictions to.

    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return os.path.join(dirname, os.path.basename(path))


def imsave(path, img, rescale):
    '''
    '''
    if rescale:
        img = (img * 255).astype(np.uint8)
    tiffsave(path, img)


def save_oof(outdir, data_dict, delimiter=','):
    '''write a set of pairs to a .csv file.

    '''
    with open(os.path.join(outdir, 'oof_predictions.csv'), 'w') as fout:
        for key, val in data_dict.items():
            fout.write(
                delimiter.join([str(key), '{:1.2f}'.format(val)]) + '\n')


def predict_flat(model, image, mask):
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



def predict(model, image, mask, batch_size=10, border=20, patch_size=None):
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
        return predict_flat(model, image, mask)

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
        zip(model.output_names, [np.zeros(image.shape[:-1] + (1, )), []]))

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

def process(stack_dir, outdir, model_path):
    '''
    '''
    logging.getLogger(__name__).info('Loading model from %s', model_path)
    model = load_model(model_path)
    is_multislice = model.input_shape[0][-1] >= 3
    if is_multislice:
        idx_dx = model.input_shape[0][-1] // 2

    # load stacks
    logging.getLogger(__name__).info('Processing folder: %s', stack_dir)
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

            prediction = predict(
                model, image, stack['mask'], border=50, batch_size=10)
            out_path = get_outpath(outdir, path)
            imsave(out_path, prediction['cell_pred'], rescale=True)


def main():
    '''
    '''
    try:
        args = parse()
        process(args.input, args.output, args.model)
    except Exception as err:
        logging.getLogger(__name__).error(str(err), exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    main()
