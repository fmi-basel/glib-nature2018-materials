from reader.reader import Reader
from reader.experiment_reader import plateReader
import pandas as pd
from utils.global_configuration import GlobalConfiguration
import os
import sys


class FeatureReaderPerPlate(Reader):
    '''
    Class to read in all experiment plates
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

        barcode = os.path.basename(root_path)
        plate_folders = next(os.walk(root_path))[1]

        for plate_folder in plate_folders:
            if plate_folder.startswith('TIF_OVR_MIP_SEG'):
                if self.segmentation_tag in next(os.walk(os.path.join(root_path, plate_folder)))[1]:

                    if os.path.exists(os.path.join(root_path, plate_folder, self.segmentation_tag, 'features')):
                        file_path = os.path.join(root_path, plate_folder, self.segmentation_tag, 'features')

                elif self.segmentation_tag == 'tag_1':
                    if os.path.exists(os.path.join(root_path, plate_folder, 'features')):
                        file_path = os.path.join(root_path, plate_folder, 'features')

                try:
                    feature_df = pd.read_csv(os.path.join(file_path, 'features_' + barcode + '.csv'))
                except IOError:
                    #print('No features for ' + barcode + 'with segmentation ' + self.segmentation_tag + ' found')
                    sys.exit('No features for ' + barcode + 'with segmentation ' + self.segmentation_tag + ' found')

                return feature_df


class SVMTrainingReaderPerPlate(Reader):
    '''
    Class to read in all experiment plates
    '''

    def __init__(self, segmentation_tag = None):

        global_config = GlobalConfiguration.get_instance()
        self.source_path = global_config.experiment_default['source_path']
        if segmentation_tag is not None:
            self.segmentation_tag = segmentation_tag
        else:
            self.segmentation_tag = global_config.segmentation_default['segmentation_tag']

    def read(self, root_path, svm_name):
        '''
        Function to load TIF_OVR_MIP_SEG, by default regionprops for each label map are loaded
        :param source_path: path to plate folder
        :param load_regionprops: Regionprops to load
        :return:
        '''

        barcode = os.path.basename(root_path)
        plate_folders = next(os.walk(root_path))[1]

        for plate_folder in plate_folders:
            if plate_folder.startswith('TIF_OVR_MIP_SEG'):
                file_path = None
                if self.segmentation_tag in next(os.walk(os.path.join(root_path, plate_folder)))[1]:

                    if os.path.exists(os.path.join(root_path, plate_folder, self.segmentation_tag, 'classification','training')):
                        file_path = os.path.join(root_path, plate_folder, self.segmentation_tag, 'classification','training')

                elif self.segmentation_tag == 'tag_1':
                    if os.path.exists(os.path.join(root_path, plate_folder, 'classification','training')):
                        file_path = os.path.join(root_path, plate_folder, 'classification','training')

                try:
                    training_df = pd.read_csv(os.path.join(file_path, 'svm_' + svm_name + '_' + barcode + '.csv'))
                except IOError:
                    print('No training data for ' + barcode + 'with segmentation ' + self.segmentation_tag + ' found: Use annotate function to generate training data')
                    sys.exit(1)

                return training_df