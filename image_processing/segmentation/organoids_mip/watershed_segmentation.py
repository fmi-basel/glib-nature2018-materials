import os

import numpy as np
from tqdm import tqdm

from skimage.morphology import watershed
from skimage.transform import resize
from skimage.filters import threshold_otsu

from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation

from image_processing.segmentation.organoids_mip.segmentation import Segmentation
from image_processing.segmentation.organoids_mip.filter.aspect_ratio import AspectRatioFilter
from image_processing.segmentation.organoids_mip.filter.region_prop import RegionPropsFilter
from image_processing.segmentation.organoids_mip.utils.imaging import close_segments

from image_processing.writer.tif_ovr_mip_segmentation_writer import TifOvrMipSegmentationWriter

class WatershedSegmentation(Segmentation):

    def segment(self, image_plate_df, parameters):
        '''

        :param image_plate_df:
        :param parameters:
        :return:
        '''
        downsampling = 4

        writer = TifOvrMipSegmentationWriter()

        for idx, row in tqdm(image_plate_df.iterrows(),
                             total=image_plate_df.shape[0],
                             desc='Images'):

            file_path = os.path.join(row.file_path, row.file_name)
            img, original_shape = self.imread(row.image, downsampling)

            segmentation = self._segment_single_well(
                img
            ).astype(np.uint16)

            segmentation = resize(segmentation,
                                  original_shape, preserve_range=True,
                                  order=0, mode='reflect',
                                  anti_aliasing=True).astype(np.uint16)

            # Fill holes
            segmentation = close_segments(segmentation)

            # Apply filter functions
            filters = [RegionPropsFilter(), AspectRatioFilter()]
            for filter in filters:
                segmentation = filter.filter(segmentation, parameters)

            # Save to disc
            writer.write({'label': segmentation}, file_path)

    def _segment_single_well(self, image):
        '''

        :param image:
        '''
        sigma = 5
        suppression_size = 21
        min_distance = 3  # local distance maxima threshold

        gaussian_filter(image, sigma=sigma, output=image)

        mask = image >= threshold_otsu(image)

        distance_map = distance_transform_edt(mask)
        maxima = maximum_filter(distance_map, size=suppression_size) == distance_map

        # Filter out maxima below threshold
        maxima[distance_map < min_distance] = False

        # Filter out close-by maxima of identical height
        fused_maxima = binary_dilation(maxima, iterations=int((suppression_size - 1) / 2))

        # label each seed and suppress local plateaus
        markers = label(fused_maxima)[0] * maxima

        segmentation = watershed(-distance_map,
                                 markers=markers,
                                 mask=mask)
        return segmentation
