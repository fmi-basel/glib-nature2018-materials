from image_processing.workflows.tasks.organoid_roi_segmentation_task import OrganoidRoiSingleCellSegmentationTask
from image_processing.workflows.tasks.organoid_roi_segmentation_task import OrganoidNucleusAndCytosolMaskGenerationTask
from image_processing.workflows.tasks.organoid_roi_feature_extraction_task import OrganoidCellCountEstimatorTask
from image_processing.workflows.tasks.organoid_roi_feature_extraction_task import OrganoidSingleCellFeatureExtractionTask


class OrganoidSingleCellProcessingWorkflow():
    def run(self, plate_filter=None, well_filter=None):
        '''
        Function to segment nuclei in organoid stacks and extract their statistics.

        :return:
        '''
        ### (1) Segment single cells on each indivual z-stack plane
        OrganoidRoiSingleCellSegmentationTask().run(
            plate_filter=plate_filter, well_filter=well_filter)

        ### (2) Combine single cell mask with organoid mask to generate nuclei/cytosol masks.
        OrganoidNucleusAndCytosolMaskGenerationTask().run(
            plate_filter=plate_filter, well_filter=well_filter)

        # ### (3) Extract Features
        OrganoidSingleCellFeatureExtractionTask().run(
            plate_filter=plate_filter, well_filter=well_filter)
        OrganoidCellCountEstimatorTask().run(
            plate_filter=plate_filter, well_filter=well_filter)


class OrganoidSingleCellFeatureExtractionWorkflow():
    def run(self, plate_filter=None, well_filter=None):
        '''
        Function to segment nuclei in organoid stacks and extract their statistics.

        :return:
        '''
        # ### Extract Features
        OrganoidSingleCellFeatureExtractionTask().run(
            plate_filter=plate_filter, well_filter=well_filter)
        OrganoidCellCountEstimatorTask().run(
            plate_filter=plate_filter, well_filter=well_filter)
