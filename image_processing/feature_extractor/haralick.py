import numpy as np
import pandas as pd
import mahotas
import time

from image_processing.feature_extractor.extractor import FeatureExtractor
from image_processing.utils.global_configuration import GlobalConfiguration

class HaralickFeatures(FeatureExtractor):

    def __init__(self, distances=None, rescale=True):

        self.rescale = rescale
        if distances is not None:
            self.distances = distances
        else:
            global_config = GlobalConfiguration.get_instance()
            self.distances = [int(d) for d in global_config.haralick_default['distance'].split(',')]
            self.channels_to_process = [int(d) for d in global_config.haralick_default['channels'].split(',')]

    def extract_features(self, image_df, feature_df, regionprops):

        '''
        To calculate the Haralick features, **MeasureTexture** normalizes the
        co-occurrence matrix at the per-object level by basing the intensity
        levels of the matrix on the maximum and minimum intensity observed
        within each object. This is beneficial for images in which the maximum
        intensities of the objects vary substantially because each object will
        have the full complement of levels.

        :param image_df:
        :param feature_df:
        :param parameter
        :return:
        '''

        t = time.process_time()
        columns = ('haralick_1_angular_second_moment_',
                   'haralick_2_contrast_',
                   'haralick_3_correlation_',
                   'haralick_4_variance_',
                   'haralick_5_inversed_difference_moment_',
                   'haralick_6_sum_average_',
                   'haralick_7_sum_variance_',
                   'haralick_8_sum_entropy_',
                   'haralick_9_entropy_',
                   'haralick_10_diference_variance_',
                   'haralick_11_difference_entropy',
                   'haralick_12_information_measures_of_correlation_1_',
                   'haralick_13_information_measures_of_Correlation_1_')

        channels = image_df['channel'].loc[image_df['image_type'] == 'intensity_image'].unique()
        channels = list(set(channels).intersection(set(self.channels_to_process)))

        haralick_df = pd.DataFrame()
        for channel in channels:
            # Extract intensity features for each object
            img = image_df['image'].loc[(image_df['image_type'] == 'intensity_image') & (image_df['channel'] == channel)].iloc[0]
            haralick_list = []
            haralick_names = ['barcode', 'well', 'z_stack', 'label']
            for distance in self.distances:
                haralick_names += [n + 'C' + str(channel) + '_distance_' + str(distance) for n in columns]

            for prop in regionprops:
                # bbox(min_row, min_col, max_row, max_col)
                min_row, min_col, max_row, max_col = prop.bbox
                img_crop = np.asarray(img.crop([min_col, min_row, max_col, max_row]))
                img_haralick = img_crop.copy()

                # Calculate haralick for 8 bit [8 levels]
                img_scale = (np.ceil(img_haralick/256)).astype(np.uint8)

                # Rescale intensity
                #img_scale = rescale_intensity(img_scale)

                # Set all values outside the mask to 0 and ignore for texture calculations
                img_scale[~prop.image] = 0

                # haralick returns features in 4 directions --> Return mean of all direction, rotational invariant
                haralick_scales = [image_df['barcode'].unique()[0],
                                   image_df['well'].unique()[0],
                                   image_df['z_stack'].unique()[0],
                                   prop.label]
                for distance in self.distances:
                    try:
                        haralick_scales = haralick_scales + list(mahotas.features.haralick(img_scale, distance=distance, ignore_zeros=True, return_mean=True))
                    except:
                        print('Haralick features could not be computed')
                        haralick_scales = haralick_scales + [np.nan] * 13

                haralick_list.append(haralick_scales)

            if len(haralick_list):
                harlick_temp_df = pd.DataFrame(haralick_list, columns=haralick_names)

                if haralick_df.empty:
                    haralick_df = harlick_temp_df
                else:
                    haralick_df = haralick_df.merge(harlick_temp_df, how='inner',
                                                      on=['barcode', 'well', 'label', 'z_stack'])

        if feature_df.empty:
            feature_df = haralick_df
        elif not haralick_df.empty:
            feature_df = feature_df.merge(haralick_df, how='inner',
                             on=['barcode', 'well', 'label', 'z_stack'])

        print('\tHaralick features for plate ' + image_df['barcode'].unique()[0]
              + ' well ' + image_df['well'].unique()[0]
              + ' z ' +  str(image_df['z_stack'].unique()[0])
              +  ' computed in ' + str(time.process_time() - t) + 'sec')

        return feature_df
