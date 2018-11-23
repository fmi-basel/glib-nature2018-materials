
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import rescale
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import exposure
from scipy import ndimage

if __name__ == "__main__":

    label_path = r'D:\mayrurs\data\experiment-organizer-local\mayrurs\171130-UM-MultiPlexTC_crop\Round1\171130UM1h3\TIF_OVR_MIP_SEG\tag_1\labels\label_171130UM1h3_C04_T0002F084L02A01Z01C04.png'
    img_path = r'D:\mayrurs\data\experiment-organizer-local\mayrurs\171130-UM-MultiPlexTC_crop\Round1\171130UM1h3\TIF_OVR_MIP\171130UM1h3_171202_093821_C04_T0002F084L02A01Z01C02.tif'

    img = imread(img_path)
    label = imread(label_path).astype(np.uint16)

    # Scale
    scale_factor = 3
    #img = rescale(img, 1 / scale_factor).astype(np.uint16)
    #label = rescale(label, 1/scale_factor).astype(np.uint16)

    # Contrast stretching
    p2, p98 = np.percentile(img, (1, 99))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))





    #label_scale = rescale(label, 0.2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    labels_rgb = label2rgb(label, img_rgb, bg_label=0, alpha=0.3, image_alpha=1, kind='overlay')

    # Get centroids of labels
    bw = label > 0
    centroids = ndimage.center_of_mass(bw.astype(int), label, np.unique(label))
    label_nr = np.unique(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 5
    fontColor = (255, 255, 255)
    lineType = 2

    for centroid, label in zip(centroids[1:], label_nr[1:]):
        cv2.putText(labels_rgb,
                    str(label),
                    (int(centroid[1]), int(centroid[0])),
                    font,
                    fontScale,
                    fontColor,
                    lineType
                    )

    imsave(r'C:\Users\mayrurs\Desktop\test.png', labels_rgb)



