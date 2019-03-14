from PIL import Image
import os, re
import pandas as pd
import numpy as np
import warnings

from image_processing.reader.reader import Reader
from image_processing.utils.global_configuration import GlobalConfiguration

# Allow to handle huge image files
Image.MAX_IMAGE_PIXELS = np.inf  # we know, as we work with large data...
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class plateReader(Reader):
    '''
      Class to read in all experiment plates
      '''

    def __init__(self, source_path=None):
        if source_path is not None:
            self.source_path = source_path
        else:
            global_config = GlobalConfiguration.get_instance()
            self.source_path = global_config.experiment_default['source_path']

    def read(self):
        '''

        :return:
        '''
        # TODO --> Add option to load shrinkaged images
        source_path = self.source_path
        expr_path_level = os.path.abspath(source_path).count(os.path.sep)
        plate_list = []
        for root, dirs, files in os.walk(source_path, topdown=True):
            for dir in dirs:
                match = re.search(r"^(\d{6})(\w{2})(\d)(.{2})$", os.path.basename(dir))
                if match:
                    plate_list.append(
                        {'plate_path': root,
                         'barcode': dir})

                if expr_path_level + 2 <= os.path.abspath(root).count(os.path.sep):
                    del dirs[:]  # stop traversing after reaching 4rd sub-directory level

        if plate_list:
            return pd.DataFrame(plate_list)

class TifOvrMipReaderPerPlate(Reader):
    '''
    Class to read in all experiment plates
    '''

    def __init__(self,  shrinkage='1x'):

        self.shrinkage = 'shrinkage_' + shrinkage
        plate_reader = plateReader()
        self.plates = plate_reader.read()

    def read(self, root_path):
        '''

        :param source_path: path to plate folder
        :param weight_channel: Intensity channel which should be used to calculate weighted regionprops
        :return:
        '''
        barcode = os.path.basename(root_path)
        image_list = []

        plate_folders = next(os.walk(root_path))[1]
        for plate_folder in plate_folders:
            if plate_folder.startswith('TIF_OVR_MIP') and not 'SEG' in plate_folder:
                type = 'intensity_image'
                shrinkage = next(os.walk(os.path.join(root_path, plate_folder)))[1]
                if self.shrinkage == 'shrinkage_1x':
                    if 'shrinkage_1x' in next(os.walk(os.path.join(root_path, plate_folder)))[1]:
                        file_path = os.path.join(root_path, plate_folder, 'shrinkage_1x')
                    else:
                        file_path = os.path.join(root_path, plate_folder)
                    print('Adding plate:' + barcode)
                    image_list = self._get_images(file_path, image_list, type, shrinkage= self.shrinkage)
                elif self.shrinkage in shrinkage:
                    file_path = os.path.join(root_path, plate_folder, self.shrinkage)
                    print('Adding plate:' + barcode)
                    image_list = self._get_images(file_path, image_list, type, shrinkage= self.shrinkage)
                else:
                    continue
            else:
                continue
        if image_list:
            return pd.DataFrame(image_list)

class TifOvrMipSegReaderPerPlate(Reader):

    '''
    Function to read in stitched MIP overviews for all channels with corresponding organoids_mip per plate
    '''

    def __init__(self, segmentation_tag = None):

        global_config = GlobalConfiguration.get_instance()
        self.source_path = global_config.experiment_default['source_path']
        if segmentation_tag is not None:
            self.segmentation_tag = segmentation_tag
        else:
            self.segmentation_tag = global_config.segmentation_default['segmentation_tag']

    def read(self, root_path):
        '''
        Function to load TIF_OVR_MIP_SEG, by default regionprops for each label map are loaded
        :param source_path: path to plate folder
        :param load_regionprops: Regionprops to load
        :return:
        '''

        image_list = []
        plate_folders = next(os.walk(root_path))[1]
        for plate_folder in plate_folders:
            if plate_folder.startswith('TIF_OVR_MIP_SEG'):
                if self.segmentation_tag in next(os.walk(os.path.join(root_path, plate_folder)))[1]:
                    img_type = 'label_image'
                    if os.path.exists(os.path.join(root_path, plate_folder, self.segmentation_tag, 'labels')):
                        file_path = os.path.join(root_path, plate_folder, self.segmentation_tag, 'labels')
                        image_list = self._get_images(file_path, image_list, img_type, segmentation_tag=self.segmentation_tag)

                elif self.segmentation_tag == 'tag_1':
                    if os.path.exists(os.path.join(root_path, plate_folder,'labels')):
                        img_type = 'label_image'
                        file_path = os.path.join(root_path, plate_folder, 'labels')
                        image_list = self._get_images(file_path, image_list, img_type, segmentation_tag=self.segmentation_tag)

        if image_list:
            image_df = pd.DataFrame(image_list)
        else:
            image_df = pd.DataFrame()
        return image_df

