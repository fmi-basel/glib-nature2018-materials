import os
import pandas as pd

def read_roi_folder(folder_path, segmentation_tag='tag_1'):
    '''
    Function returns a generator which can be executed step wise with next() or completely with list()
    :param exclude_dir_names:
    :param accu:
    :return:
    '''

    if not os.path.exists(os.path.join(folder_path,'TIF_OVR_MIP_SEG')):
        print('No segmentation and roi folder found')
    elif segmentation_tag != 'tag_1':
        if not os.path.exists(os.path.join(folder_path,'TIF_OVR_MIP_SEG', segmentation_tag)):
            print('No roi folder found for specified segmentation')
    else:
        if os.path.exists(os.path.join(folder_path,'TIF_OVR_MIP_SEG', 'tag_1')):
            base_path = os.path.join(folder_path,'TIF_OVR_MIP_SEG', 'tag_1','crop')
        else:
            base_path = os.path.join(folder_path, 'TIF_OVR_MIP_SEG','crop')

        crop_list = []
        if os.path.exists(base_path):
            well_list = os.listdir(base_path)
            for well in well_list:
                crop_list.append({'folder_path' : base_path, 'well' : well})
            return pd.DataFrame(crop_list, columns=['folder_path','well'])