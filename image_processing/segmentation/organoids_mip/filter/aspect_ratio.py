import numpy as np

from mahotas.labeled import relabel, remove_regions
from skimage.measure import regionprops

from image_processing.segmentation.organoids_mip.filter.base import Filter

class AspectRatioFilter(Filter):

    def __init__(self):
        self.filter_name = 'Aspect_ratio_filter'

    def filter(self, label_image,  parameters):

        props = regionprops(label_image, coordinates='rc' )
        aspect_ratio_tresh = parameters.aspect_ratio

        minor_axis = np.array([a.minor_axis_length for a in props])
        major_axis = np.array([e.major_axis_length for e in props])

        aspect_ratio = np.divide(major_axis, minor_axis)

        # Filter out objects
        to_filter = list((aspect_ratio > aspect_ratio_tresh))

        # Keep trivial label 0: background
        to_filter.insert(0, False)
        to_filter_idx = np.where(to_filter)

        relabeled, nr_obj = relabel(remove_regions(label_image, to_filter_idx), inplace=True)

        return relabeled.astype(np.uint16)