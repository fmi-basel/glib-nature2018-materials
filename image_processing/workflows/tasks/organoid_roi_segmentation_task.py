import os

from image_processing.segmentation.organoids_roi.neural_network_segmentation_sp import NeuralNetworkSegmentationSP
from image_processing.reader.experiment_reader import plateReader
from image_processing.reader.roi_reader import read_roi_folder

class OrganoidRoiSegmentationTask():

    def run(self, plate_filter=None, well_filter=None):
        '''
        Task to run segmentation on individual planes of a z-stack for the provided plates with provides method in segmentation_parameters
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
        segmentation = NeuralNetworkSegmentationSP()

        for idx, plate in plates_df.iterrows():
            wells_with_crops_df = read_roi_folder(os.path.join(plate.plate_path, plate.barcode))
            if not wells_with_crops_df.empty:
                if well_filter is None:
                    well_filter = list(set(wells_with_crops_df['well']))
                else:
                    well_filter = list(set(wells_with_crops_df['well']).intersection(set(well_filter)))
                for idx, well_crop in wells_with_crops_df.iterrows():
                    if well_crop.well in well_filter:
                        roi_path = os.path.join(well_crop.folder_path, well_crop.well, 'roi')
                        if os.path.exists(roi_path):
                            segmentation.segment(roi_path)


