import configparser
import os

class GlobalConfiguration():
    """
    A global configuration class, implemented as singleton.
    NOTE: This class is NOT thread-safe!
    """
    config_file_path = ""
    config = configparser.ConfigParser()

    def __init__(self, config_file=None):

        if len(GlobalConfiguration.config.sections()) != 0 and config_file is not None:
            raise RuntimeError("Global configuration already initialized!")

        if config_file is not None:
            GlobalConfiguration.config_file_path = config_file
            GlobalConfiguration.config.read(config_file)


    @staticmethod
    def get_instance():
        if len(GlobalConfiguration.config.sections()) == 0:
            raise RuntimeError("Global configuration not yet initialized!")
        return GlobalConfiguration()

    @property
    def zernike_default(self):
        return GlobalConfiguration.config["zernike_default"]

    @property
    def haralick_default(self):
        return GlobalConfiguration.config["haralick_default"]

    @property
    def intensity_default(self):
        return GlobalConfiguration.config['intensity_default']

    @property
    def superpixel_default(self):
        return GlobalConfiguration.config['superpixel_default']

    @property
    def extractors_default(self):
        return GlobalConfiguration.config["extractors_default"]

    #@property
    #def experiment_default(self):
    #   return GlobalConfiguration.config['experiment_default']

    @property
    def segmentation_default(self):
        return GlobalConfiguration.config['segmentation_default']

    @property
    def acquisition_default(self):
        return GlobalConfiguration.config['acquisition_default']

    @property
    def experiment_default(self):
        return {'source_path' : os.path.dirname(self.config_file_path)}

    @property
    def multiprocessing_default(self):
        return GlobalConfiguration.config['multiprocessing_default']



