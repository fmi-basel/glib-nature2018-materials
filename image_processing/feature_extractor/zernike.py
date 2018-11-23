import numpy as np
import time
import pandas as pd
import mahotas
import cv2
from skimage import img_as_ubyte

from image_processing.feature_extractor.extractor import FeatureExtractor
from image_processing.utils.global_configuration import GlobalConfiguration

from image_processing.feature_extractor.utils.smallest_enclosing_circle import make_circle

class ZernikeFeatures(FeatureExtractor):

    def __init__(self, degree=None):
        if degree is not None:
            self.distances = degree
        else:
            global_config = GlobalConfiguration.get_instance()
            self.degree = int(global_config.zernike_default['degree'])

    def extract_features(self, image_df, feature_df, regionprops):

        t = time.process_time()
        properties = []
        indices = []

        # Get feature names
        zernike_names = ['barcode', 'well', 'z_stack', ]
        for n in range(self.degree + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    zernike_names.append('zernike_feature_degree_' + str(n) + '_' + str(l))

        zernike_meta = [image_df['barcode'].unique()[0],
                    image_df['well'].unique()[0],
                    image_df['z_stack'].unique()[0]]

        for prop in regionprops:

            # initialize the outline image, find the outermost
            # contours (the outline) of the organoid
            label_image =  img_as_ubyte(prop.image.copy())

            # pad the image with extra black pixels to ensure the
            # edges of the organoid are not up against the borders
            # of the image
            label_image = cv2.copyMakeBorder(label_image, 15, 15, 15, 15,
                                       cv2.BORDER_CONSTANT, value=0)

            (img, contours, hierarchy) = cv2.findContours(label_image, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE)
            # Only keep largest contour
            cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            contours = np.vstack(cnt).squeeze()

            # Compute smallest enclosing circle
            (center_x, center_y, radius)= make_circle(contours)

            # Get zernike moments
            zernike = zernike_meta + list(mahotas.features.zernike_moments(label_image, radius, degree=self.degree, cm=(center_x, center_y)))

            properties.append(zernike)
            indices.append(prop.label)

        if indices:
            indices = pd.Index(indices, name='label')
            zernike_df = pd.DataFrame(properties, index = indices, columns=zernike_names).reset_index()

        if feature_df.empty:
            feature_df = zernike_df
        else:
            feature_df = feature_df.merge(zernike_df, how='inner',
                             on=['barcode', 'well', 'label', 'z_stack'])

        print('\tZernike features for plate ' + image_df['barcode'].unique()[0]
              + ' well ' + image_df['well'].unique()[0]
              + ' z ' + str(image_df['z_stack'].unique()[0])
              + ' computed in ' + str(time.process_time() - t) + 'sec')

        return feature_df
