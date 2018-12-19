import abc
import six
import os
import logging

from image_processing.workflows.tasks.base import Task

from image_processing.reader.experiment_reader import plateReader
from image_processing.reader.roi_reader import read_roi_folder

from image_processing.utils.dataframe_utils import filter_by

from image_processing.segmentation.organoids_roi.neural_network_segmentation import NeuralNetworkSegmentationSP
from image_processing.segmentation.organoids_roi.neural_network_segmentation import NeuralNetworkSegmentationSC
from image_processing.segmentation.organoids_roi.nucleus_and_cytosol_segmentation import NucleusAndCytosolWatershed


@six.add_metaclass(abc.ABCMeta)
class _BaseRoiSegmentationTask(Task):
    '''Base ROI segmentation Task. Fetches experiment and calls segmentation method
    on each of it's wells.

    NOTE Implementations have to set self.segmentation_methos.
    See OrganoidRoiSegmentationTask as an example.

    '''

    @property
    @abc.abstractmethod
    def segmentation_method(self):
        pass

    def __init__(self, *args, **kwargs):
        '''
        '''
        super(_BaseRoiSegmentationTask, self).__init__(*args, **kwargs)
        self._segmenter = None

    def _init_segmentation_method(self):
        '''instantiates the segmentation method once at the
        beginning of run().

        '''
        self._segmenter = self.segmentation_method()

    def segmentation_call(self, *args, **kwargs):
        '''calls the segmentation method on a well.

        '''
        return self._segmenter.segment(*args, **kwargs)

    def run(self, plate_filter=None, well_filter=None):
        '''
        '''
        plates_df = plateReader().read()

        if plate_filter is not None:
            plates_df = filter_by(plates_df, 'barcode', plate_filter)

        self._init_segmentation_method()

        for _, plate in plates_df.iterrows():
            wells_with_crops_df = read_roi_folder(
                os.path.join(plate.plate_path, plate.barcode))

            # TODO Should read_roi_folder raise instead of returning None?
            if wells_with_crops_df is None or wells_with_crops_df.empty:
                logging.getLogger(__name__).warn(
                    'Could not read crops for barcode: %s at %s',
                    plate.barcode, plate.plate_path)
                continue

            if well_filter is not None:
                wells_with_crops_df = filter_by(wells_with_crops_df, 'well',
                                                well_filter)

            for _, well_crop in wells_with_crops_df.iterrows():
                roi_path = os.path.join(well_crop.folder_path, well_crop.well,
                                        'roi')
                out_path = os.path.join(well_crop.folder_path, well_crop.well,
                                        'roi_pred')

                if os.path.exists(roi_path):
                    self.segmentation_call(roi_path, out_path)


class OrganoidRoiSegmentationTask(_BaseRoiSegmentationTask):
    '''Task to run segmentation on individual planes of a z-stack for the
    provided plates with provides method in segmentation_parameters.

    NOTE Segmentation only possible if crop/roi folder exists.

    '''

    segmentation_method = NeuralNetworkSegmentationSP


class OrganoidRoiSingleCellSegmentationTask(_BaseRoiSegmentationTask):
    '''Task to run segmentation on individual planes of a z-stack for the
    provided plates with provides method in segmentation_parameters.

    NOTE Segmentation only possible if crop/roi folder exists.

    '''

    segmentation_method = NeuralNetworkSegmentationSC


class OrganoidNucleusAndCytosolMaskGenerationTask(_BaseRoiSegmentationTask):
    '''Task to run segmentation on individual planes of a z-stack for the
    provided plates with provides method in segmentation_parameters.

    NOTE Segmentation only possible if crop/roi folder exists.

    '''

    segmentation_method = NucleusAndCytosolWatershed
