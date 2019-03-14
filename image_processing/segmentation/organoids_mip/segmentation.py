import abc
import six
import numpy as np
import cv2

from skimage.measure import block_reduce
from image_processing.dlutils.preprocessing.normalization import standardize

from mahotas.labeled import relabel, remove_regions_where


@six.add_metaclass(abc.ABCMeta)
class Segmentation():
    @abc.abstractmethod
    def segment(self, image_plate_df, segmentation_tag):
        pass

    def imread(self, image, downsampling):
        '''Load image and standardize.
        '''
        img = np.asarray(image)
        original_shape = img.shape
        img = block_reduce(
            img, tuple([
                downsampling,
            ] * img.ndim), func=np.mean)
        return standardize(img, min_scale=1000.), original_shape

    def imsave(self, path, img, rescale):
        '''
        '''
        if rescale:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)
