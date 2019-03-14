import pandas as pd

from image_processing.segmentation.organoids_mip.neural_network_segmentation_mip import NeuralNetworkSegmentationMIP
from image_processing.segmentation.organoids_mip.watershed_segmentation import WatershedSegmentation

from image_processing.workflows.tasks.organoid_mip_segmentation_task import OrganoidMipSegmentationTask
from image_processing.workflows.tasks.organoid_roi_segmentation_task import OrganoidRoiSegmentationTask
from image_processing.workflows.tasks.organoid_mip_feature_extraction_task import OrganoidMipFeatureExtractionTask

from image_processing.workflows.tasks.organoid_roi_feature_extraction_task import OrganoidRoiFeatureExtractionTask

class OrganoidMipProcessingWorkflow():

    def run(self):
        '''
        Function to segment and extract features for organoids on stitched mip overviews
        :return:
        '''

        ### (1) Segment organoids on mip stitched overviews

        # By default a watershed after Otsu thresholding is used for small objects
        segmentation_parameters = pd.DataFrame([
        {'barcode': '181230AA001', 'segmentation_method': WatershedSegmentation(), 'network': None,
             'area': (1000, 1200000), 'circularity': (0.0, 1), 'solidity': (0.6, 1.0), 'aspect_ratio': 3, 'channel': 4},
       ])

        # The neural network based segmentation can be run on the test dataset by using the following lines:
        #segmentation_parameters = pd.DataFrame([
        #{'barcode': '181230AA001', 'segmentation_method': NeuralNetworkSegmentationMIP(), 'network': 'v_1',
        #'area': (1000, 1200000), 'circularity': (0.0, 1), 'solidity': (0.6, 1.0), 'aspect_ratio': 3, 'channel': 4},
        #])

        # Run segmentation workflows
        OrganoidMipSegmentationTask().run(segmentation_parameters)

        ### (2) Extract features for segmented objects
        OrganoidMipFeatureExtractionTask().run()

class OrganoidSinglePlaneProcessingWorkflow():

    def run(self, plate_filter=None, well_filter=None):
        '''
        Function to segment organoid stacks and extract 3d features for organoids
        :return:
        '''

        ### (1) Segment organoids on each indivual z-stack plane
        OrganoidRoiSegmentationTask().run()

        ### (2) Extract 3d features over z-stack segmentatino
        OrganoidRoiFeatureExtractionTask().run()

