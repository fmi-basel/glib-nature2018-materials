import os
import cv2

import numpy as np

from image_processing.writer.writer import Writer
from image_processing.utils.global_configuration import GlobalConfiguration

class TifOvrMipSegFeaturesWriter(Writer):

    '''
    Writer
    '''

    def __init__(self):
        global_config = GlobalConfiguration.get_instance()
        self.segmentation = global_config.segmentation_default['segmentation_tag']

    def write(self, data_frame, target_file_name):
        '''
        Function to write csv for plate features
        :param data_frame:
        :param target_file_name:
        :return:
        '''
        output_folder = os.path.dirname(target_file_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        data_frame.to_csv(target_file_name, index=False)

class TifOvrMipSegmentationWriter(Writer):

    def _imsave(self, path, img, rescale = False):
        '''
        Writer
        '''
        if rescale:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)

    def __init__(self):
        global_config = GlobalConfiguration.get_instance()
        self.segmentation = global_config.segmentation_default['segmentation_tag']

    def write(self, segmentation, target_file_name):

        '''
        Writer to write organoid mip segmentations into a TIF_OVR_MIP_SEG folder
        '''

        folder_path = target_file_name.split('TIF_OVR_MIP', 1)[0]
        for key, val in segmentation.items():
            # TODO --> Re-name naming in segmentation function in dl-utlils
            if key == 'nuclei_segm':
                key = 'label'
            elif key == 'cell_pred':
                key = 'prediction_organoid'
            elif key ==  'border_pred':
                key = 'prediction_border'
            # TODO --; Allow for more than one tag
            file_name = os.path.basename(target_file_name)
            file_name_tmp = os.path.splitext(file_name)[0]
            if 'prediction' in key:
                dir = os.path.join(folder_path, 'TIF_OVR_MIP_SEG', 'predictions')
            elif 'label' in key:
                dir = os.path.join(folder_path, 'TIF_OVR_MIP_SEG' , 'labels')

            if not os.path.exists(dir):
                os.makedirs(dir)

            self._imsave(os.path.join(dir, key + '_' + file_name_tmp + '.tif'), val, rescale='prediction' in key)



