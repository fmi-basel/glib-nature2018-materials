import numpy as np
import pandas as pd
import time
import scipy

from scipy import ndimage
from skimage.segmentation import slic
from skimage.exposure import rescale_intensity
from skimage.segmentation import relabel_sequential

from skimage.morphology import remove_small_objects

from image_processing.feature_extractor.extractor import FeatureExtractor
from image_processing.utils.global_configuration import GlobalConfiguration

OUTSIDE = 0

class IntensityFeatureSuperpixel(FeatureExtractor):

    def __init__(self, superpixel_default=None):

        if superpixel_default is None:
            global_config = GlobalConfiguration.get_instance()
            superpixel_default = global_config.superpixel_default

        self.cell_area = int(superpixel_default['cell_area'])
        self.number_superpixel = int(superpixel_default['number_superpixel'])
        self.thresh = int(superpixel_default['thresh'])
        self.rescale_range = tuple([int(d) for d in global_config.superpixel_default['rescale_range'].split(',')])
        self.superpixel_channel = [int(d) for d in global_config.superpixel_default['channel'].split(',')]

    def estimate_segment_count(self, area, cell_area):
        '''

        :param area:
        :param cell_area:
        :return:
        '''
        return np.max([area // cell_area, 1])

    def extract_features(self, image_df, feature_df, regionprops):

        '''
        :param image_df:
        :param feature_df:
        :param parameter:
        :return:
        '''

        t = time.process_time()
        channels = image_df['channel'].loc[image_df['image_type'] == 'intensity_image'].unique()
        barcode = image_df['barcode'].unique()[0]
        well = image_df['well'].unique()[0]
        z_stack = image_df['z_stack'].unique()[0]

        intensity_df = pd.DataFrame()
        channels = set(channels).intersection(set(self.superpixel_channel))
        for channel in channels:
            try:
                img = image_df['image'].loc[(image_df['image_type'] == 'intensity_image') & (image_df['channel'] == channel)].iloc[0]
                feat_list = []
                for prop in regionprops:
                    min_row, min_col, max_row, max_col = prop.bbox
                    mask = prop.image
                    img_crop = np.asarray(img.crop([min_col, min_row, max_col, max_row]))
                    rescaled = rescale_intensity(img_crop, in_range=self.rescale_range)
                    rescaled[np.logical_not(mask)] = 0
                    spx = slic(rescaled,
                               n_segments=self.estimate_segment_count(np.prod(img_crop.shape), self.cell_area), compactness=0.5,
                               multichannel=False,
                               enforce_connectivity=True, max_iter=10)
                    # Only consider superpixels within mask
                    spx[np.logical_not(mask)] = OUTSIDE
                    # Remove small cutted edge superpixel
                    remove_small_objects(spx, min_size=self.cell_area // 4, in_place=True)
                    spx = relabel_sequential(spx)[0]
                    nr_superpixel = spx.max()

                    if nr_superpixel == 0:
                        # If object area is smaller than one superpixel take object mean
                        pixel_mean = np.mean(img_crop[mask])
                        above_thresh = 0
                    else:
                        index = list(range(1, nr_superpixel + 1))
                        means = scipy.ndimage.mean(img_crop, spx, index=index)
                        means_sorted = np.sort(means)[::-1]
                        # Number of 'positive' cells
                        above_thresh = np.sum(means > self.thresh)
                        if nr_superpixel >= self.number_superpixel:
                            pixel_mean = np.mean(means_sorted[:self.number_superpixel+1])
                        else:
                            pixel_mean = np.mean(means_sorted)

                    feat_list.append({
                         'barcode': barcode,
                         'well': well,
                         'z_stack': z_stack,
                         'label': prop.label,
                         'intensity_superpixel_mean_c' + str(channel): pixel_mean,
                         'intensity_superpixel_number_c' + str(channel): nr_superpixel,
                         'intensity_superpixel_above_c' + str(channel): above_thresh})

            except IOError:
                print('One of the channel images could not be loaded')

            if len(feat_list):
                int_temp_df = pd.DataFrame(feat_list).fillna(0)

                if intensity_df.empty:
                    intensity_df = int_temp_df
                else:
                    intensity_df = intensity_df.merge(int_temp_df, how = 'inner', on = ['barcode','well','label', 'z_stack'])

        if feature_df.empty:
            feature_df = intensity_df
        elif not intensity_df.empty:
            feature_df = feature_df.merge(intensity_df, how='inner',
                             on=[ 'barcode', 'well', 'label', 'z_stack'])

        print('\tSuperpixel intensity features for plate ' + image_df['barcode'].unique()[0]
              + ' well ' + image_df['well'].unique()[0]
              + ' z ' +  str(image_df['z_stack'].unique()[0])
              +  ' computed in ' + str(time.process_time() - t) + 'sec')

        return feature_df
