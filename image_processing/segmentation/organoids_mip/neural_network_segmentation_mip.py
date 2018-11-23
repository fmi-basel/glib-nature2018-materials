import os
import numpy as np
import logging
from tqdm import tqdm

from skimage.transform import resize

from dlutils.models import load_model
from dlutils.prediction import predict_complete
from dlutils.postprocessing.watershed import segment_nuclei

from image_processing.segmentation.organoids_mip.segmentation import Segmentation
from image_processing.segmentation.organoids_mip.utils.imaging import close_segments
from image_processing.segmentation.organoids_mip.filter.aspect_ratio import AspectRatioFilter
from image_processing.segmentation.organoids_mip.filter.region_prop import RegionPropsFilter

from image_processing.writer.tif_ovr_mip_segmentation_writer import TifOvrMipSegmentationWriter

from image_processing.utils.global_configuration import GlobalConfiguration

# logger format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S')
logger = logging.getLogger(__name__)

# disable tensorflow clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NeuralNetworkSegmentationMIP(Segmentation):

    def __init__(self, segmentation_tag=None):
        global_config = GlobalConfiguration.get_instance()
        self.source_path = global_config.experiment_default['source_path']

        if not segmentation_tag is None:
            self.segmentation_tag = segmentation_tag
        else:
            self.segmentation_tag = global_config.segmentation_default['segmentation_tag']

    def segment(self, image_plate_df, parameters):
        '''

        :param image_plate_df:
        :param parameters:
        :return:
        '''

        model_base_path = os.path.join(os.path.dirname(__file__), 'networks')
        model_path = os.path.join(model_base_path, parameters.network, 'model.h5')
        logger.info('Loading model from {} ...'.format(model_path))
        model = load_model(model_path)
        logger.info('successful!')

        downsampling = 4

        writer = TifOvrMipSegmentationWriter()

        for idx, row in tqdm(image_plate_df.iterrows(),
                             total=image_plate_df.shape[0],
                             desc='Images'):

            file_path = os.path.join(row.file_path, row.file_name)
            img, original_shape = self.imread(row.image, downsampling)

            segmentation = predict_complete(model, img, border=50, batch_size=4, patch_size=(1024,1024))

            segmentation['nuclei_segm'] = segment_nuclei(
                segmentation['cell_pred'],
                segmentation['border_pred'],
                threshold=0.5,
                upper_threshold=0.8,
                # watershed_line=True,
            ).astype(np.uint16)

            segmentation['nuclei_segm'] = resize(segmentation['nuclei_segm'],
                                              original_shape, preserve_range=True,
                                               order=0, mode='reflect',
                                               anti_aliasing=True).astype(np.uint16)

            label_image = segmentation['nuclei_segm']

            # Fill holes
            label_image = close_segments(label_image)

            # Apply filter functions
            filters = [RegionPropsFilter(), AspectRatioFilter()]
            for filter in filters:
                label_image = filter.filter(label_image, parameters)

            segmentation['nuclei_segm'] = label_image

            # Save to disc
            writer.write(segmentation, file_path)
