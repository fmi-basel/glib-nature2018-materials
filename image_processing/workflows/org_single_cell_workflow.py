from image_processing.workflows.tasks.organoid_roi_segmentation_task import OrganoidRoiSingleCellSegmentationTask


class OrganoidSingleCellProcessingWorkflow():
    def run(self, plate_filter=None, well_filter=None):
        '''
        Function to segment nuclei in organoid stacks and extract their statistics.

        :return:
        '''

        ### (1) Segment single cells on each indivual z-stack plane
        OrganoidRoiSingleCellSegmentationTask().run()

        ### (2) Combine single cell mask with organoid mask to generate nuclei/cytosol masks.

        # ### (2) Extract Features
        # TODO OrganoidSingleCellFeatureExtractionTask().run()
