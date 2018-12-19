import logging
import os
import pandas

from image_processing.feature_extractor.extractor import FeatureExtractor


class CellCountEstimator(FeatureExtractor):
    '''estimates cell count per organoid and writes result into a csv.

    '''

    expected_slices_per_nucleus = 2
    out_name = 'cell_count_features.csv'

    def extract_features(self, roi_pred_dir):
        '''calculate cell count estimate for the given folder.

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

        in_path = os.path.join(out_dir, 'single_cell_features.csv')
        if not os.path.exists(in_path):
            raise RuntimeError(
                'Could not find single cell features at {}'.format(in_path))

        # read single cell counts and count cells per organoid.
        cell_features = pandas.read_csv(in_path)
        cell_count = cell_features.groupby(
            ['barcode', 'well', 'label']).size().reset_index(name='counts')

        # correct for multiple counts.
        cell_count.counts /= self.expected_slices_per_nucleus

        # write output
        cell_count.to_csv(out_path)
