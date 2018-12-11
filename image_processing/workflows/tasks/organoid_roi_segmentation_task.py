import os

from .base import Task

from image_processing.segmentation.organoids_roi.neural_network_segmentation_sp import NeuralNetworkSegmentationSP
from image_processing.reader.experiment_reader import plateReader
from image_processing.reader.roi_reader import read_roi_folder


class _BaseRoiSegmentationTask(Task):
    '''Base ROI segmentation Task. Fetches experiment and calls segmentation method
    on each of it's wells.

    NOTE Implementations have to set self.segmentation_methos.
    See OrganoidRoiSegmentationTask as an example.

    '''

    # TODO consider using abc to force self.segmentation_method to exist.

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

        # TODO could this be simplified with filter_by?
        if plate_filter is None:
            plate_filter = list(plates_df['barcode'])
        else:
            plate_filter = list(
                set(list(plates_df['barcode'])).intersection(plate_filter))

        plates_df = plates_df.loc[plates_df['barcode'].isin(plate_filter)]

        self._init_segmentation_method()

        for idx, plate in plates_df.iterrows():
            wells_with_crops_df = read_roi_folder(
                os.path.join(plate.plate_path, plate.barcode))
            if not wells_with_crops_df.empty:
                # TODO could this be simplified with filter_by?
                if well_filter is None:
                    well_filter = list(set(wells_with_crops_df['well']))
                else:
                    well_filter = list(
                        set(wells_with_crops_df['well']).intersection(
                            set(well_filter)))
                for idx, well_crop in wells_with_crops_df.iterrows():
                    if well_crop.well in well_filter:
                        roi_path = os.path.join(well_crop.folder_path,
                                                well_crop.well, 'roi')
                        if os.path.exists(roi_path):
                            self.segmentation_call(roi_path)


class OrganoidRoiSegmentationTask(_BaseRoiSegmentationTask):
    '''Task to run segmentation on individual planes of a z-stack for the
    provided plates with provides method in segmentation_parameters.

    NOTE Segmentation only possible if crop --> roi folder exists

    '''

    segmentation_method = NeuralNetworkSegmentationSP
