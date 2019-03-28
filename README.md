# Experiment Organizer

Organizer for indexing and processing microscope experiments.

The application allows the indexing of image and meta data from microscope experiments within an Sqlite database. Hereby, images and annotations are stored within a spatial index that allows a fast access and spatial operations.

After indexing an experiment, it is possible to apply user-defined workflows for data processing and analysing. Two pre-defined workflows are available for linking organoids over acquisition rounds or cells over z-planes.

## Requirements

The code was written for Python 3.6 and is not tested with Python 2.7.

## Installation

Under Windows and Linux you can install the Python package using pip:

    pip install https://github.com/fmi-basel/glib-nature2018-materials/archive/organoid_linking_algorithm.zip

Hereby, the pre-build Sqlite extension mod_spatialite-4.3.0a will be installed automatically.
If you need a newer version of mod_spatialite or the pre-build binaries do not work on your system please refer to https://www.gaia-gis.it/gaia-sins/ for further information.

## Usage
Following, a simple example to index an experiment and analyze the data using two pre-defined workflows:

    from experiment_organizer import ExperimentOrganizer

    experiment_path = '/path/to/my/experiment'
    with ExperimentOrganizer(experiment_path) as experiment_organizer:
        # index images and segmentations from an experiment folder
        experiment_organizer.build_index(
          load_images=True,
          load_segmentations=True,
          load_segmentation_features=True,
        )
        # run workflow for linking organoids over acquisition rounds
        experiment_organizer.link_organoids()
        # run workflow for linking cells over z-planes
        experiment_organizer.link_cells()

## License

MIT License

Copyright (c) 2019 Dario Vischi (FMI)

You should have received a copy of the MIT License along with the source code.
If not, see https://opensource.org/licenses/MIT
