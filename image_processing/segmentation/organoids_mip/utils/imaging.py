from scipy.ndimage.measurements import  find_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import relabel_sequential
import numpy as np

def close_segments(segmentation):
    '''
    Fill holes: Added UM

    :param segmentation:
    :return:
    '''
    segmentation = relabel_sequential(segmentation)[0].astype(np.uint16) # Return 16bit
    bboxes = find_objects(segmentation)

    for component_label, bbox in zip(range(1, len(bboxes)), bboxes):

        segmentation_part = segmentation[bbox]
        mask = segmentation_part == component_label

        mask_filled = binary_fill_holes(mask.squeeze())

        #_, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
        #axarr[0].imshow(mask.squeeze())
        #axarr[1].imshow(mask_filled.squeeze())

        segmentation[bbox][mask_filled] = component_label


    return segmentation
