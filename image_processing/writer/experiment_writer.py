import os
import numpy as np
import cv2
import re

from image_processing.reader.experiment_reader import TifOvrMipReader
from image_processing.writer.writer import Writer
from image_processing.utils.global_configuration import GlobalConfiguration

class TifOvrMipWriter(Writer):

    '''
    Writer to write out

    '''

    def _imsave(path, img, rescale = False):
        '''
        '''
        if rescale:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)

    def __init__(self):
        global_config = GlobalConfiguration.get_instance()
        self.segmentation = global_config.segmentation_default['segmentation_tag']

    def _move_files(self, group):
        '''
        subfunction to move files
        :return:
        '''
        for idx, file in group.iterrows():
            file_path_source = os.path.join(file.file_path, file.file_name)
            file_path_target =  os.path.join(file.file_path, 'shrinkage_1x', file.file_name)
            os.rename(file_path_source, file_path_target)

    def write(self, shrinkage = '1x'):

        '''
        Function to manuplate size and write tif Ovr
        '''

        ovr_reader = TifOvrMipReader()
        mip_ovr = ovr_reader.read()
        to_move = []

        scale_exp = re.compile("\d{1,2}")
        scale_factor = int(scale_exp.search(shrinkage).group(0))

        for barcode, group in mip_ovr.groupby('barcode'):
            folder_path = group['file_path'].unique()[0]
            shrinkage_folder = os.listdir(os.path.dirname(folder_path))
            if not 'shrinkage_' + shrinkage in shrinkage_folder:
                if shrinkage == '1x' and not 'shrinkage_1x' in folder_path:
                    # Move files into subfolder
                    print(barcode + ': Moving raw TIP_MIP_OVR into shrinkage_1x subfolder ')
                    self._move_files(group)
                elif shrinkage != '1x':
                    # Generate shrinkage
                    if not 'shrinkage_1x' in folder_path:
                        folder_path_target = os.path.join(folder_path, 'shrinkage_' + shrinkage)
                    else:
                        base = os.path.dirname(folder_path)
                        folder_path_target = os.path.join(base, 'shrinkage_' + shrinkage)
                    if not os.path.exists(folder_path_target):
                        os.mkdir(folder_path_target)
                    print(barcode + ': Shrinking ')
                    for idx, file in group.iterrows():
                        image = file.image
                        file_path = os.path.join(folder_path_target, file.file_name)
                        width, height = image.size
                        image_scaled = image.resize((int(width/scale_factor),int(height/scale_factor)))
                        image_scaled.save(file_path)
                        if not 'shrinkage_1x' in folder_path:
                            to_move.append({'barcode' : barcode, 'file_path': file.file_path,'file_name': file.file_name})

        # Clean up 'shrinkage_1x folder
        mip_ovr = None
        #if to_move:
        #    print(barcode + ': Moving raw TIP_MIP_OVR into shrinkage_1x subfolder ')
        #    move_group = pd.DataFrame(to_move, columns = ['barcode', 'file_path','file_name'])
        #   self._move_files(move_group)

class TifOvrMipSegFeaturesWriter(Writer):

    '''
    Writer to write out

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



class TifOvrCropWriter(Writer):

    '''
    Writer to write out

    '''

    def __init__(self):
        global_config = GlobalConfiguration.get_instance()
        self.segmentation = global_config.segmentation_default['segmentation_tag']

    def write(self, crop, target_file_name):
        '''
        Function to write csv for plate features
        :param data_frame:
        :param target_file_name:
        :return:
        '''
        output_folder = os.path.dirname(target_file_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(target_file_name, crop)









