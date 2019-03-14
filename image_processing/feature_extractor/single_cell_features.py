import logging
import os
from glob import glob

import pandas
import numpy as np
from skimage.external.tifffile import imread

from image_processing.feature_extractor.extractor import FeatureExtractor
from image_processing.reader.stackreader import _parse_from_filename


def calculate_nucleus_features(nuclei_segm):
    '''returns features for each nucleus in the segmentation.

    '''
    assert nuclei_segm.ndim == 2

    labels = np.unique(nuclei_segm)
    assert labels[0] == 0
    labels = labels[1:]

    # if the segmentation is inconsistent, we escape.
    if len(labels) == 0:
        return

    # check for identical labels
    features = {}
    for cell_id in labels:
        features['cell'] = cell_id
        features['nucleus_area'] = np.sum(nuclei_segm == cell_id)

        yield features


class SingleCellFeatureExtractor(FeatureExtractor):
    '''calculates single cell features from nuclei segmentation.

    '''

    out_name = 'single_cell_features.csv'
    _pattern = os.path.join('nuclei_segm', '*tif')

    def extract_features(self, roi_pred_dir):
        '''calculate features and save dataframe for 3d image stacks.

        NOTE Requires single_cell_features to be precalculated as input.
        See OrganoidSingleCellFeatureExtractionTask

        '''

        out_dir = os.path.join(os.path.dirname(roi_pred_dir), 'features')
        out_path = os.path.join(out_dir, self.out_name)
        if os.path.exists(out_path):
            logging.getLogger(__name__).info(
                '%s is already processed. skipping.', roi_pred_dir)
            return

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        features = []
        for path in glob(os.path.join(roi_pred_dir, self._pattern)):

            meta_info = _parse_from_filename(os.path.basename(path))

            for cell_features in calculate_nucleus_features(imread(path)):
                features.append({**meta_info, **cell_features})

        # convert and write.
        features = pandas.DataFrame(features)
        features.to_csv(out_path)
