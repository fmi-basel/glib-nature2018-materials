import os
import pandas as pd

from image_processing.workflows.tasks.base import Task
from image_processing.reader.experiment_reader import TifOvrMipReaderPerPlate
from image_processing.reader.experiment_reader import plateReader

class OrganoidMipSegmentationTask(Task):

    def __init__(self):
        pass

    def run(self, segmentation_parameters):
        '''
        Task to run segmentation on provided plates with provides method in segmentation_parameters
         segmentation_parameters = [
        {'barcode': '171130UM1h1', 'segmentation_method': NeuralNetworkSegmentation(), 'network': None, 'size': (100,1000) , 'eccentricity': (0,1), 'channel' : 1},

        :param segmentation_parameters:
        :param segmentation_tag:
        :return:
        '''

        segmentations = pd.DataFrame(segmentation_parameters)

        # Load mips for segmentation
        image_reader = TifOvrMipReaderPerPlate()
        plates_df = plateReader().read()

        for idx, seg_group in segmentations.iterrows():

            barcode = seg_group.barcode
            if barcode in plates_df['barcode'].tolist():
                seg_method = seg_group.segmentation_method
                seg_channel = seg_group.channel
                plate_group = plates_df[plates_df['barcode'] == barcode].iloc[0]
                image_df = image_reader.read(os.path.join(plate_group.plate_path, plate_group.barcode))
                seg_group_image = image_df[image_df['channel'] == seg_channel]
                seg_method.segment(seg_group_image,seg_group)
            else:
                print('Barcode not in experiment')
