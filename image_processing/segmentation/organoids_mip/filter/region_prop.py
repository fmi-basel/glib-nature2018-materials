import numpy as np

from mahotas.labeled import relabel, remove_regions
from skimage.measure import regionprops

from image_processing.segmentation.organoids_mip.filter.base import Filter

class RegionPropsFilter(Filter):

    def __init__(self):
        self.filter_name = 'region_props_filter'

    def filter(self, label_image, parameters):

        props = regionprops(label_image, coordinates='rc')
        areas = np.array([a.area for a in props])
        peri = np.array([p.perimeter for p in props])
        solidity = np.array([s.solidity for s in props])

        # Circularity (Fiji definition)
        cirularity = np.multiply(4 * np.pi, np.divide(areas, np.multiply(peri, peri)))

        min_area, max_area = parameters.area
        (min_circularity, max_circularity) = parameters.circularity
        (min_solidity, max_solidity) = parameters.solidity

        # Apply filter
        to_filter = list((areas < min_area) | (areas > max_area) |
                    (cirularity < min_circularity) | (cirularity > max_circularity) |
                    (solidity < min_solidity) | (solidity > max_solidity))

        #sizes = mh.labeled.labeled_size(label_image)
        #too_big = np.where(sizes < 30000)

        # Keep trivial label 0: background
        to_filter.insert(0, False)
        to_filter_idx = np.where(to_filter)

        relabeled, nr_obj = relabel(remove_regions(label_image, to_filter_idx), inplace=True)

        return relabeled.astype(np.uint16)