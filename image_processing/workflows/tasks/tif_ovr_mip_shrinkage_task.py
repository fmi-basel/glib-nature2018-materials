import os
import numpy as np
import cv2
import re

from image_processing.reader.experiment_reader import plateReader
from image_processing.reader.experiment_reader import TifOvrMipReader, TifOvrReaderPerPlate
from image_processing.utils.global_configuration import GlobalConfiguration


class TifOvrMipShrinkageTask():

    '''
    Writer to write out

    '''

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
            target_folder = os.path.dirname(file_path_target)
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)

            # TODO --> Add movement option
            try:
                os.rename(file_path_source, file_path_target)
            except:
                print('Could not move file' + file_path_source)

    def run(self, shrinkage='1x'):

        '''
        Function to manipulate size and write tif Ovr
        '''

        ovr_reader = TifOvrMipReader()
        mip_ovr = ovr_reader.read()
        to_move = []

        scale_exp = re.compile("\d{1,2}")
        scale_factor = int(scale_exp.search(shrinkage).group(0))

        for barcode, group in mip_ovr.groupby('barcode'):
            folder_path = group['file_path'].unique()[0]
            shrinkage_folder = os.listdir(os.path.dirname(folder_path))
            # Clean up folder
            if not 'shrinkage_' + shrinkage in shrinkage_folder:
                if shrinkage == '1x' and not 'shrinkage_1x' in folder_path:
                    # TODO --> Add automatic file movement
                    # Move files into subfolder
                    #print(barcode + ': Moving raw TIP_MIP_OVR into shrinkage_1x subfolder ')
                    #self._move_files(group)
                    pass
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

                            # Clean up memory
                            image.close()
                            image_scaled.save(file_path)

                            # Clean up memory
                            image_scaled.close()

                            if not 'shrinkage_1x' in folder_path:
                                to_move.append({'barcode' : barcode, 'file_path': file.file_path,'file_name': file.file_name})
                    else:
                        print('shrinkage for ' + barcode + ' already exists')


class TifOvrShrinkageTask():

    '''
    Writer to write out

    '''

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
            target_folder = os.path.dirname(file_path_target)
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)

            # TODO --> Add movement option
            try:
                os.rename(file_path_source, file_path_target)
            except:
                print('Could not move file' + file_path_source)

    def run(self, shrinkage='1x', plate_filter=None):

        '''
        Function to manipulate size and write tif Ovr
        '''

        ovr_reader = TifOvrReaderPerPlate()
        plates_df = plateReader().read()

        if plate_filter is None:
            plate_filter = list(plates_df['barcode'])
        else:
            # Remove filter values which are not in the experiment folder
            plate_filter = list(set(list(plates_df['barcode'])).intersection(plate_filter))

        plates_df = plates_df.loc[plates_df['barcode'].isin(plate_filter)]
        to_move = []

        scale_exp = re.compile("\d{1,2}")
        scale_factor = int(scale_exp.search(shrinkage).group(0))

        for idx, row in plates_df.iterrows():

            mip_ovr = ovr_reader.read(os.path.join(row.plate_path, row.barcode))
            folder_path = mip_ovr['file_path'].unique()[0]
            shrinkage_folder = os.listdir(os.path.dirname(folder_path))
            # Clean up folder
            if not 'shrinkage_' + shrinkage in shrinkage_folder:
                if shrinkage == '1x' and not 'shrinkage_1x' in folder_path:
                    # TODO --> Add automatic file movement
                    # Move files into subfolder
                    #print(barcode + ': Moving raw TIP_MIP_OVR into shrinkage_1x subfolder ')
                    #self._move_files(group)
                    pass
                elif shrinkage != '1x':
                    # Generate shrinkage
                    if not 'shrinkage_1x' in folder_path:
                        folder_path_target = os.path.join(folder_path, 'shrinkage_' + shrinkage)
                    else:
                        base = os.path.dirname(folder_path)
                        folder_path_target = os.path.join(base, 'shrinkage_' + shrinkage)
                    if not os.path.exists(folder_path_target):
                        os.mkdir(folder_path_target)
                        print(row.barcode + ': Shrinking ')
                        for idx, file in mip_ovr.iterrows():
                            image = file.image
                            file_path = os.path.join(folder_path_target, file.file_name)
                            width, height = image.size
                            image_scaled = image.resize((int(width/scale_factor),int(height/scale_factor)))

                            # Clean up memory
                            image.close()
                            image_scaled.save(file_path)

                            # Clean up memory
                            image_scaled.close()

                            if not 'shrinkage_1x' in folder_path:
                                to_move.append({'barcode' : file.barcode, 'file_path': file.file_path,'file_name': file.file_name})
                    else:
                        pass
                        #print('shrinkage for ' + file.barcode + ' already exists')



if __name__ == "__main__":

    # Load configuration file for the experiment
    GlobalConfiguration(config_file=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gliberal\Users\mayrurs\181011-UM-MultiPlexYap\config_file.ini')

    from image_processing.workflows.tasks.tif_ovr_mip_shrinkage_task import TifOvrShrinkageTask

    #Task to shrink MIP_OVR_TIF
    shrinkage_task = TifOvrShrinkageTask()
    shrinkage_task.run(shrinkage='3x',plate_filter=['181011UM1f3','181011UM1h3','181011UM1f4','181011UM1h4','181011UM1f5'])





