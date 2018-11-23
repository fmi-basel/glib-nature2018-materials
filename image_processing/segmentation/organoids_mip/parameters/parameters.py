### Pre-defined parameters: For each fixation time a segmentation method and filter criteria are set:
from image_processing.segmentation.organoids_mip.neural_network_segmentation import NeuralNetworkSegmentation
from image_processing.segmentation.organoids_mip.watershed_segmentation import WatershedSegmentation

barcode_meta = {
    0: {'segmentation_method': WatershedSegmentation(),
                    'network': None,
                    'area': (1000, 1200000),
                    'circularity': (0.0, 1),
                    'solidity': (0.6, 1.0),
                    'aspect_ratio': 3,
                    'channel': 4},

    24 : {'segmentation_method': WatershedSegmentation(),
                        'network': None,
                        'area': (1000, 1200000),
                        'circularity': (0.0, 1),
                        'solidity': (0.6, 1.0),
                        'aspect_ratio': 3,
                        'channel': 4},

    48 : {'segmentation_method': WatershedSegmentation(),
                            'network': None,
                            'area': (1000, 1200000),
                            'circularity': (0.0, 1),
                            'solidity': (0.6, 1.0),
                            'aspect_ratio': 3,
                            'channel': 4},

    72 : {'segmentation_method': WatershedSegmentation(),
                                'network': None,
                                'area': (1000, 1200000),
                                'circularity': (0.0, 1),
                                'solidity': (0.6, 1.0),
                                'aspect_ratio': 3,
                                'channel': 4},

    84: {'segmentation_method': WatershedSegmentation(),
         'network': None,
         'area': (1000, 1200000),
         'circularity': (0.0, 1),
         'solidity': (0.6, 1.0),
         'aspect_ratio': 3,
         'channel': 4},

    96: {'segmentation_method': NeuralNetworkSegmentation(),
                    'network': 'v_3',
                    'area': (1000, 1200000),
                    'circularity': (0.0, 1),
                    'solidity': (0.6, 1.0),
                    'aspect_ratio': 3,
                    'channel': 4},

    120 : {'segmentation_method': NeuralNetworkSegmentation(),
             'network': 'v_3',
             'area': (1000, 1200000),
             'circularity': (0.0, 1),
             'solidity': (0.6, 1.0),
             'aspect_ratio': 3,
             'channel': 4}}



