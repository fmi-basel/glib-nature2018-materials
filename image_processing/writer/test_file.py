
from skimage.io import imread
from skimage.transform import rescale
import cv2

if __name__ == "__main__":

    img_path = r'D:\mayrurs\data\experiment-organizer-local\mayrurs\171130-UM-MultiPlexTC_crop\Round1\171130UM1h3\TIF_OVR_MIP_SEG\tag_1\labels\label_171130UM1h3_C03_T0001F081L01A01Z01C04.png'
    label = r'D:\mayrurs\data\experiment-organizer-local\mayrurs\171130-UM-MultiPlexTC_crop\Round1\171130UM1h3\TIF_OVR_MIP\171130UM1h3_171202_093821_C04_T0002F084L02A01Z01C02.tif'

    img = imread(img_path)
    label = imread(label)

    # = rescale(img, 0.2)
    #label_scale = rescale(label, 0.2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    labels_rgb = label2rgb(label, img_rgb)

   # img_rgb = cv2.cvtColor(image_scale, cv2.COLOR_GRAY2RGB)
   # combined_rgb = cv2.addWeighted(img_rgb, 0.4, labels_rgb)

    # Get centroids of labels
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
