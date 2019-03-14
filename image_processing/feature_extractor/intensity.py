import numpy as np
import pandas as pd
import time

from scipy import ndimage
from skimage import img_as_float
from statsmodels.robust.scale import mad

from image_processing.utils.global_configuration import GlobalConfiguration
from image_processing.feature_extractor.extractor import FeatureExtractor

class IntensityFeatures(FeatureExtractor):

    def __init__(self, pixel_size = None):
        if pixel_size is not None:
            self.pixel_size = pixel_size
        else:
            global_config = GlobalConfiguration.get_instance()
            self.pixel_size = float(global_config.intensity_default['pixel_size'])

    def _mass_displacement(self, img_crop, bw, pixel_size):

        '''

        :param img:
        :param bw:
        :return:
        '''
        # Only calculate mass displacement within object
        img = img_crop.copy()
        img[bw] = 0

        img = img_crop.copy()
        bw_x = np.sum(np.multiply(np.array(list(range(1, bw.shape[1] + 1))), np.array(np.sum(bw, axis=0)))) / np.sum(
            np.array(list(range(1, bw.shape[1] + 1))))
        bw_y = np.sum(np.multiply(np.array(list(range(1, bw.shape[0] + 1))), np.array(np.sum(bw, axis=1)))) / np.sum(
            np.array(list(range(1, bw.shape[0] + 1))))
        img_x = np.sum(np.multiply(np.array(list(range(1, img.shape[1] + 1))), np.array(np.sum(img, axis=0)))) / np.sum(
            np.array(list(range(1, img.shape[1] + 1))))
        img_y = np.sum(
            np.multiply(np.array(list(range(1, img.shape[0] + 1))), np.array(np.sum(img_crop, axis=1)))) / np.sum(
            np.array(list(range(1, img.shape[0] + 1))))

        return np.sqrt((bw_x - img_x) ** 2 + (bw_y - img_y) ** 2) * pixel_size

    def extract_features(self, image_df, feature_df, regionprops):

        '''
        LowerQuartileIntensity:* The intensity value of the pixel for which 25% of the pixels in the object have lower values.
        UpperQuartileIntensity:* The intensity value of the pixel for which 75% of the pixels in the object have lower values.
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
        for channel in channels:
            img = image_df['image'].loc[(image_df['image_type'] == 'intensity_image') & (image_df['channel'] == channel)].iloc[0]
            feat_list = []
            for prop in regionprops:

                # bbox(min_row, min_col, max_row, max_col)
                min_row, min_col, max_row, max_col = prop.bbox
                img_crop = np.asarray(img.crop([min_col, min_row, max_col, max_row]))
                object_filled = img_crop[prop.image]

                # The mass displacement is the distance between the center
                # of mass of the binary image and of the intensity image. The
                # center of mass is the average X or Y for the binary image
                # and the sum of X or Y * intensity / integrated intensity
                lmask = prop.image

                # Weight image in [0,1] range
                img_range = img_as_float(img_crop.copy())

                labels = prop.image.astype(int)
                lindexes = 1
                nobjects = 1
                mesh_y, mesh_x = np.mgrid[ 0:labels.shape[0], 0:labels.shape[1]]
                mesh_x = mesh_x[lmask]
                mesh_y = mesh_y[lmask]
                limg = img_range[lmask]
                llabels = labels[lmask]
                cm_x = ndimage.mean(mesh_x, labels=llabels, index=lindexes)
                cm_y = ndimage.mean(mesh_y, labels=llabels, index=lindexes)
                i_x = ndimage.sum(mesh_x * limg, llabels, index=lindexes)
                i_y = ndimage.sum(mesh_y * limg, llabels, index=lindexes)
                integrated_intensity = ndimage.sum(limg, llabels, lindexes)

                cmi_x = np.zeros((nobjects,))
                cmi_y = np.zeros((nobjects,))

                cmi_x[lindexes - 1] = i_x / integrated_intensity
                cmi_y[lindexes - 1] = i_y / integrated_intensity

                diff_x = cm_x - cmi_x[lindexes - 1]
                diff_y = cm_y - cmi_y[lindexes - 1]

                mass_displacement = np.sqrt(diff_x * diff_x + diff_y * diff_y)

                #
                # Calculate standard features and collect
                #
                feat_list.append({'barcode': barcode,
                     'well': well,
                     'z_stack': z_stack,
                     'label': prop.label,
                     'intensity_sum_c' + str(channel): np.sum(object_filled),
                     'intensity_mean_c' + str(channel): np.mean(object_filled),
                     'intensity_std_c' + str(channel): np.std(object_filled),
                     'intensity_median_c' + str(channel): np.median(object_filled),
                     'intensity_mad_c' + str(channel): mad(object_filled),
                     'intensity_max_c' + str(channel): np.amax(object_filled),
                     'intensity_min_c' + str(channel): np.amin(object_filled),
                     'intensity_mass_displacement_c' + str(channel): mass_displacement,
                     'intenstiy_lower_quartile_c' + str(channel): np.percentile(object_filled, 25),
                     'intenstiy_upper_quartile_c' + str(channel): np.percentile(object_filled, 75)})

            if len(feat_list):
                int_temp_df = pd.DataFrame(feat_list)

                if intensity_df.empty:
                    intensity_df = int_temp_df
                else:
                    intensity_df = intensity_df.merge(int_temp_df, how = 'inner', on = ['barcode','well','label', 'z_stack'])

        if feature_df.empty:
            feature_df = intensity_df
        elif not intensity_df.empty:
            feature_df = feature_df.merge(intensity_df, how='inner',
                             on=[ 'barcode', 'well', 'label', 'z_stack'])

        print('\tIntensity features for plate ' + image_df['barcode'].unique()[0]
              + ' well ' + image_df['well'].unique()[0]
              + ' z ' +  str(image_df['z_stack'].unique()[0])
              +  ' computed in ' + str(time.process_time() - t) + 'sec')

        return feature_df
