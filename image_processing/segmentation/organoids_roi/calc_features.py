'''Calculate features for single organoids in 3d.
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import logging

import pandas

from tqdm import tqdm

from features.stackreader import StackGenerator
from features.features import segment
from features.features import calc_base_features
from features.features import calc_derived_features

# logger format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S')


def parse():
    '''parse command line arguments.
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--image_stacks', help='folder with image stacks', required=True)
    parser.add_argument(
        '--segm_stacks', help='folder with probability stacks', required=True)
    return parser.parse_args()


def make_dir_if_necessary(path):
    '''
    '''
    dirname = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dirname):
        logging.getLogger(__name__).info(
            'Creating folder at %s to save features', dirname)
        os.makedirs(dirname)


def process(img_dir, segm_dir, out_path, spacing):
    '''calculate features for a folder of stacks and corresponding segmentations.

    '''
    features = []
    focus_threshold = 0.25

    for stack in tqdm(
            StackGenerator(img_dir=img_dir, segm_dir=segm_dir),
            desc='Processing stacks',
            ncols=80):

        segm, focus_prob = segment(
            stack['segm_stack'],
            image=stack['image_stack'],
            threshold=0.5,
            focus_threshold=focus_threshold)
        features.append({
            **{key: stack[key]
               for key in ['barcode', 'well', 'label']},
            **{
                'n_planes': len(stack['image_stack']),
                'n_infocus_planes': (focus_prob > focus_threshold).sum(),
                'n_segmented_planes': (stack['segm_stack'].max(axis=(1, 2))).sum()
            },
            **calc_base_features(segm, stack['mask'], spacing)
        })

    if len(features) == 0:
        raise RuntimeError('Could not process any stack in {} and {}.'.format(
            img_dir, segm_dir))

    features = pandas.DataFrame(features)
    features = calc_derived_features(features)
    logging.getLogger(__name__).info('Calculated features of %d objects',
                                     len(features))
    logging.getLogger(__name__).info('Saving features to %s', out_path)

    make_dir_if_necessary(out_path)
    features.to_csv(out_path)


def main():
    '''
    '''
    try:
        args = parse()

        # Consider making 'spacing' an input argument
        spacing = (3.0, 0.1625, 0.1625)

        process(
            args.image_stacks,
            args.segm_stacks,
            os.path.abspath(
                os.path.join(args.segm_stacks, '..', 'features',
                             'features.csv')),
            spacing=spacing)
    except Exception as err:
        logging.getLogger(__name__).error(str(err), exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    main()
