import os

from image_processing.workflows.workflows import OrganoidMipProcessingWorkflow
from image_processing.utils.global_configuration import GlobalConfiguration

if __name__ == "__main__":

    config_file = dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'config_file.ini')
    # Load default configurations for the project
    GlobalConfiguration(config_file=config_file)

    # Segment and extract features for organoids on mip overviews
    OrganoidMipProcessingWorkflow().run()
