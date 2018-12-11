import re
import os
from glob import glob
import logging

import pandas
import numpy as np
from skimage.external.tifffile import imread


def _parse_from_filename(filename):
    '''parse information about experiment and slice from file name.

    Expected filename structure:
        obj100_180823UM1f0_C03_T0001F001L01A01Z01C04.tif
        -> label=100, barcode=180823UM1f0, well=C03, zplane=01

    '''
    pattern_str = 'obj(?P<label>[0-9]+)_(?P<barcode>.[^_]*).*_(?:(?P<well_row>[A-Z])(?P<well_col>[0-9]+))_(?:.*(?P<zplane>(?:Z)[0-9]+)|(?P<mask>mask)).*'
    matches = re.match(pattern_str, filename)
    if matches is None:
        raise RuntimeError(
            'Could not match filename: {} with pattern {}'.format(
                filename, pattern_str))
    parsed = {
        key: matches.group(key)
        for key in ['label', 'barcode', 'zplane', 'mask']
    }

    parsed['well'] = '{}{:02}'.format(
        matches.group('well_row'), int(matches.group('well_col')))
    parsed['is_mask'] = matches.group('mask') is not None
    if parsed['zplane'] is not None:
        parsed['zplane'] = int(parsed['zplane'][1:])

    for key in [
            'label',
    ]:
        parsed[key] = int(parsed[key])

    return parsed


def _split_mask_and_stack(stack):
    '''splits mask from image stack for crops that were saved
    with the old preprocessing.

    Parameters
    ----------
    stack : array-like, shape=(ZCYX)
        image and mask stack combined. The mask is expected
        to be in channel 2 and invariant along Z.

    Returns
    -------
    mask : array-like, shape=(ZYX)
        mask of the crop.
    stack : array-like, shape=(ZYX)
        image stack.

    '''
    assert stack.shape[1] == 2
    assert stack.ndim == 4
    mask = stack[0, 1, ...]
    return mask, stack[:, 0, ...]


def collect_from(img_dir, segm_dir=None, parser_fn=None, pattern=None):
    '''collects all paths from a given directory that match the provided pattern.

    Parameters
    ----------
    img_dir : string
        path to image crop directory.
    segm_dir : string
        path to probability map crop directory.
    parser_fn : function
        function to parse meta information from file name.

    Returns
    -------
    df : pandas.DataFrame
        data frame of collected paths.

    '''
    if parser_fn is None:
        parser_fn = _parse_from_filename

    if pattern is None:
        pattern = ['*C04.tif', '*mask.tif']

    paths = []
    for pt in pattern:
        paths.extend(sorted(glob(os.path.join(img_dir, pt))))

    if segm_dir is None:

        def _find_segm(*args):
            return None
    else:

        def _find_segm(path):
            '''silently returns None if no segmentation is found.

            '''
            path = os.path.join(segm_dir, os.path.basename(path))
            if os.path.exists(path):
                return path
            return None

    return pandas.DataFrame([{
        'img_path': path,
        'segm_path': _find_segm(path),
        **parser_fn(os.path.basename(path))
    } for path in paths])


def read_channel(paths, source_dir):
    '''reads stack based on matching file names.

    Notes
    -----
    paths are expected to be sorted w.r.t. Z-axis.

    '''
    channel_stack = [imread(os.path.join(source_dir, os.path.basename(path)))
                     for path in paths]
    return np.asarray(channel_stack).squeeze()


class StackGenerator(object):
    '''read crop stacks from a given folder and group them by object.

    '''

    def __init__(self,
                 df=None,
                 img_dir=None,
                 segm_dir=None,
                 groupby=None,
                 **kwargs):
        '''
        '''
        if df is None and img_dir is None:
            raise ValueError('Either df or img_dir have to be set!')

        logging.getLogger(__name__).debug('Reading image stacks from %s',
                                          img_dir)
        logging.getLogger(__name__).debug('Readimg predictions from %s',
                                          segm_dir)

        if df is not None:
            self.df = df
        else:
            self.df = collect_from(img_dir, segm_dir, **kwargs)
            self.with_segm = segm_dir is not None

        if groupby is None:
            groupby = ['barcode', 'well', 'label']

        self.groupby = groupby

    def __len__(self):
        '''
        '''
        return len(self.df.groupby(self.groupby))

    def __iter__(self):
        '''
        '''
        IMG_KEY = 'img_path'
        SEGM_KEY = 'segm_path'

        def _read_stack(group, key):
            return np.vstack([
                imread(x[key])[None, ...] for _, x in group.iterrows()
                if x[key] is not None
            ]).squeeze()

        for group_key, group in self.df.groupby(self.groupby):

            # sort by z-plane and separate mask from rest.
            vals = {
                key: val.sort_values(by='zplane')
                for key, val in group.groupby('is_mask')
            }

            stack = dict(zip(self.groupby, group_key))
            try:
                stack['image_paths'] = [
                    val[IMG_KEY] for _, val in vals[False].iterrows()
                ]

                stack['image_stack'] = _read_stack(vals[False], IMG_KEY)

                try:
                    stack['mask'] = _read_stack(vals[True], IMG_KEY)
                except KeyError as err:
                    if stack['image_stack'].shape[1] == 2 and \
                       stack['image_stack'].ndim == 4:

                        stack['mask'], stack[
                            'image_stack'] = _split_mask_and_stack(
                                stack['image_stack'])
                    else:
                        raise RuntimeError(
                            'Could not find mask for {}'.format(group_key))

                # make sure mask is binary
                stack['mask'] = stack['mask'] >= 1

                # and check for matching shape of mask and image stack.
                assert stack['mask'].shape == stack['image_stack'].shape[1:]

                if self.with_segm:
                    stack['segm_stack'] = _read_stack(vals[False], SEGM_KEY)
                else:
                    stack['segm_stack'] = None
            except (ValueError, KeyError) as err:
                logging.getLogger(__name__).error(
                    'Could not process stack of {}. Skipping...'.format(
                        group_key))
                logging.getLogger(__name__).error('Error message: ' + str(err))
                continue

            yield stack
