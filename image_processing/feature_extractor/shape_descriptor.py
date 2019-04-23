import pandas as pd
import time

from image_processing.feature_extractor.extractor import FeatureExtractor
from image_processing.utils.global_configuration import GlobalConfiguration

class ShapeDescriptor(FeatureExtractor):
    '''
    Class to calculate morphological features of organoids on MIPs
    '''

    def __init__(self):

        global_config = GlobalConfiguration.get_instance()
        self.weight_channel = global_config.segmentation_default['weight_channel']

    def extract_features(self, image_df, feature_df, regionprops):

        t = time.process_time()

        barcode = image_df['barcode'].unique()[0]
        well = image_df['well'].unique()[0]
        z_stack = image_df['z_stack'].unique()[0]
        feat_list = []
        for d in regionprops:
            if not d._intensity_image is None:
                weighted_centroid = d.weighted_centroid
            else:
                weighted_centroid = (None, None)

            feat_list.append(
            {'barcode': barcode,
             'well': well,
             'z_stack': z_stack,
             'label': d.label,
             'shape_descriptor_area' : d.area,
             'shape_descriptor_centroid_y' : d.centroid[0],
             'shape_descriptor_centroid_x': d.centroid[1],
             'shape_descriptor_convex_area': d.convex_area,
             'shape_descriptor_eccentricity': d.eccentricity,
             'shape_descriptor_equivalent_diameter': d.equivalent_diameter,
             'shape_descriptor_euler_number': d.euler_number,
             'shape_descriptor_extent': d.extent,
             'shape_descriptor_filled_area': d.filled_area,
             'shape_descriptor_major_axis_length': d.major_axis_length,
             'shape_descriptor_minor_axis_length': d.minor_axis_length,
             'shape_descriptor_orientation': d.orientation,
             'shape_descriptor_perimeter': d.perimeter,
             'shape_descriptor_solidity': d.solidity,
             'shape_descriptor_weighted_centroid_y_' + self.weight_channel: weighted_centroid[0],
             'shape_descriptor_weighted_centroid_x_' + self.weight_channel: weighted_centroid[1],
             })

        if len(feat_list):
            shape_df = pd.DataFrame(feat_list)
            if feature_df.empty:
                feature_df = shape_df
            else:
                feature_df = feature_df.merge(shape_df, how='inner',
                                                  on=['barcode', 'well', 'label', 'z_stack'])

        print('\tShape descriptors for plate ' + image_df['barcode'].unique()[0]
              + ' well ' + image_df['well'].unique()[0]
              + ' z ' + str(image_df['z_stack'].unique()[0])
              + ' computed in ' + str(time.process_time() - t) + 'sec')

        return feature_df

