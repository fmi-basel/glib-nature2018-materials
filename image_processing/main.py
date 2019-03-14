from image_processing.workflows.tasks.organoid_mip_segmentation_task import OrganoidMipSegmentationTask
#from segmentation.organoids_mip.neural_network_segmentation import NeuralNetworkSegmentation

from image_processing.utils.global_configuration import GlobalConfiguration
#from segmentation.organoids_mip.neural_network_segmentation import NeuralNetworkSegmentation
#from segmentation.organoids_mip.watershed_segmentation import WatershedSegmentation
#import pandas as pd
#from workflows.tasks.organoid_mip_feature_extraction_task import OrganoidMipFeatureExtractionTask
#from workflows.tasks.tif_ovr_mip_shrinkage_task import TifOvrMipShrinkageTask
from image_processing.workflows.tasks.rgb_overview_creation_task import RGBOverviewCreationTask

if __name__ == "__main__":

    # Load configuration file for the experiment
    GlobalConfiguration(config_file= r'\\tungsten-nas.fmi.ch\tungsten\scratch\gliberal\Users\mayrurs\171130-UM-MultiPlexTC_linkage\config_file.ini')

    # Extract Features
    #feature_extractor = OrganoidMipFeatureExtractionTask(extractors='ShapeDescriptor,IntensityFeatures')
    #feature_extractor.run(filter=['180823UM2f0','180823UM2f1','180823UM2h1','180823UM2f2','180823UM2h2','180823UM2f3','180823UM2h3','180823UM2f4','180823UM2h4','180823UM2f5','180823UM2f6'], overwrite=False, append_existing=False)


    ### DEBUG:
    # = np.asarray(Image.open(r'W:\Users\mayrurs\180823-UM-MultiplexingTC-Lgr5DTR\Round1\180823UM1f4\TIF_OVR_MIP_SEG\tag_1\label\label_180823UM1f4_180908_222224_C04_T0002F001L02A01Z01C04.tif'))
    #segmentation_parameters = pd.DataFrame([
    #{'barcode': '171130UM1f2', 'segmentation_method': NeuralNetworkSegmentation(), 'network': 'v_3',
    # 'area': (1000, 1200000), 'circularity': (0.0, 1), 'solidity': (0.6, 1.0), 'aspect_ratio': 3, 'channel': 4}])

    #segmentation_task = OrganoidMipSegmentationTask()
    #segmentation_task.run(segmentation_parameters)


    ##plt.imshow(label_image)
    #plt.show()
    #filter = RegionPropsFilter()
    #relabel = filter.filter(label_image, parameters.iloc[0])

    #plt.imshow(relabel)
    #plt.show()

    # Extract Features
    #feature_extractor = OrganoidMipFeatureExtractionTask(extractors=None)
    #feature_extractor.run(filter=None, overwrite = False, append_existing = False)

    # Task to shrink MIP_OVR_TIF
    #shrinkage_task = TifOvrMipShrinkageTask()
    #shrinkage_task.run(shrinkage = '3x')

    # Task to generate RGB overviews
    rgb_creation_task = RGBOverviewCreationTask()
    rgb_creation_task.run(rgb_overview_shrinkage = '3x', shrinkage_to_load='3x',
                          rgb_channel_composition='3,2,1',
                          rgb_channel_clipping="(50,2500),(50,2000),(50,2000)",
                          )

    # Segment organoids
    #segmentation_parameters = [
     #   {'barcode': '180823UM3h4', 'segmentation_method':  NeuralNetworkSegmentation(), 'network': 'v_3',
     #    'size': (100, 100000), 'eccentricity': (0, 1),  'channel': 4},
    #]
    #segmentation_task = OrganoidMipSegmentationTask()
   # segmentation_task.run(segmentation_parameters)


