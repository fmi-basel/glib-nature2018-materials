import os
import tqdm

from multiprocessing import Pool

from image_processing.utils.global_configuration import GlobalConfiguration
from image_processing.reader.experiment_reader import plateReader
from image_processing.reader.roi_reader import read_roi_folder
from image_processing.feature_extractor.shape_descriptor_3d import ShapeDescriptor3d

class OrganoidRoiFeatureExtractionTask():

    '''
    Workflow to extract 3d features for cropped image stacks
    '''
    def __init__(self, nr_of_cores=None):

        if not nr_of_cores is None:
            self.nr_of_cores = nr_of_cores
        else:
            global_config = GlobalConfiguration.get_instance()
            self.nr_of_cores = int(global_config.multiprocessing_default['nr_of_cores'])

    def run(self, plate_filter=None, well_filter=None):
        '''
        Task to run segmentation on individual planes of a z-stack
        ! Segmentation only possible if crop --> roi folder exists
        :param segmentation_parameters:
        :param segmentation_tag:
        :return:
        '''

        plates_df = plateReader().read()
        if plate_filter is None:
            plate_filter = list(plates_df['barcode'])
        else:
            plate_filter = list(set(list(plates_df['barcode'])).intersection(plate_filter))

        plates_df = plates_df.loc[plates_df['barcode'].isin(plate_filter)]
        feature_extractor = ShapeDescriptor3d()

        for idx, plate in plates_df.iterrows():
            wells_with_crops_df = read_roi_folder(os.path.join(plate.plate_path, plate.barcode))
            if not wells_with_crops_df.empty:
                if well_filter is None:
                    well_filter = list(set(wells_with_crops_df['well']))
                else:
                    well_filter = list(set(wells_with_crops_df['well']).intersection(set(well_filter)))

                crop_paths = [os.path.join(well_crop.folder_path, well_crop.well) for idx, well_crop in wells_with_crops_df.iterrows() if well_crop.well in well_filter]
                pred_paths = [os.path.join(pred, 'roi_pred') for pred in crop_paths if os.path.exists(os.path.join(pred, 'roi_pred'))]

                if len(pred_paths) < self.nr_of_cores:
                    nr_of_cores = len(pred_paths)
                else:
                    nr_of_cores = self.nr_of_cores
                with Pool(nr_of_cores) as pool:
                    list(tqdm.tqdm(pool.imap(feature_extractor.extract_features, pred_paths), total=len(pred_paths),
                            desc='Extracting 3d features for well:'))

                #for _, well_crop in wells_with_crops_df.iterrows():
                #   if well_crop.well in well_filter:
                #       pred_path = os.path.join(well_crop.folder_path, well_crop.well, 'roi_pred')
                #       if os.path.exists(pred_path):
                #           feature_extractor.extract_features(pred_path)
