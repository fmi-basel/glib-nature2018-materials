import os

from image_processing.workflows import OrganoidSingleCellProcessingWorkflow
from image_processing.utils.global_configuration import GlobalConfiguration

if __name__ == "__main__":

    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'data', 'config_file.ini')
    # Load default configurations for the project
    GlobalConfiguration(config_file=config_file)

    # Segment and extract features for individual cells
    OrganoidSingleCellProcessingWorkflow().run()
