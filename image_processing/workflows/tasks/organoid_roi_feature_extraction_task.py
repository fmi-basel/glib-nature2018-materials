import os
import tqdm
import abc
import logging

from multiprocessing import Pool

from image_processing.workflows.tasks.base import Task
from image_processing.utils.global_configuration import GlobalConfiguration
from image_processing.reader.experiment_reader import plateReader
from image_processing.reader.roi_reader import read_roi_folder
from image_processing.feature_extractor.shape_descriptor_3d import ShapeDescriptor3d
from image_processing.feature_extractor.cell_count_estimator import CellCountEstimator
from image_processing.feature_extractor.single_cell_features import SingleCellFeatureExtractor
from image_processing.utils.dataframe_utils import filter_by


class BaseRoiFeatureExtractionTask(Task):
    '''
    Task to extract features for cropped image stacks
    '''

    def __init__(self, nr_of_cores=None):

        if nr_of_cores is not None:
            self.nr_of_cores = nr_of_cores
        else:
            global_config = GlobalConfiguration.get_instance()
            self.nr_of_cores = int(
                global_config.multiprocessing_default['nr_of_cores'])

    @property
    @abc.abstractmethod
    def extractor_method(self):
        pass

    @property
    @abc.abstractmethod
    def description(self):
        pass

    def run(self, plate_filter=None, well_filter=None):
        '''
        Task to run segmentation on individual planes of a z-stack
        ! Segmentation only possible if crop --> roi folder exists
        :param segmentation_parameters:
        :param segmentation_tag:
        :return:
        '''

        plates_df = plateReader().read()
        if plate_filter is not None:
            plates_df = filter_by(plates_df, 'barcode', plate_filter)

        feature_extractor = self.extractor_method()

        for idx, plate in plates_df.iterrows():
            wells_with_crops_df = read_roi_folder(
                os.path.join(plate.plate_path, plate.barcode))

            if wells_with_crops_df is None or wells_with_crops_df.empty:
                logging.getLogger(__name__).warn(
                    'Could not read crops for barcode: %s at %s',
                    plate.barcode, plate.plate_path)
                continue

            if well_filter is not None:
                wells_with_crops_df = filter_by(wells_with_crops_df, 'well',
                                                well_filter)

            pred_paths = []
            for crop_path in (
                    os.path.join(well_crop.folder_path, well_crop.well)
                    for _, well_crop in wells_with_crops_df.iterrows()):

                pred_path = os.path.join(crop_path, 'roi_pred')
                if os.path.exists(pred_path):
                    pred_paths.append(pred_path)
                else:
                    logging.getLogger(__name__).warning(
                        'Could not find prediction folder for %s', crop_path)

            if len(pred_paths) <= 0:  # nothing to do.
                return

            if len(pred_paths) < self.nr_of_cores:
                nr_of_cores = len(pred_paths)
            else:
                nr_of_cores = self.nr_of_cores
            with Pool(nr_of_cores) as pool:
                list(
                    tqdm.tqdm(
                        pool.imap(feature_extractor.extract_features,
                                  pred_paths),
                        total=len(pred_paths),
                        desc=self.description))


class OrganoidRoiFeatureExtractionTask(BaseRoiFeatureExtractionTask):

    extractor_method = ShapeDescriptor3d
    description = 'Extracting 3d features for well'


class OrganoidSingleCellFeatureExtractionTask(BaseRoiFeatureExtractionTask):

    extractor_method = SingleCellFeatureExtractor
    description = 'Extracting single cell features'


class OrganoidCellCountEstimatorTask(BaseRoiFeatureExtractionTask):

    extractor_method = CellCountEstimator
    description = 'Estimating cell count per organoid'
