#!/usr/bin/env python3

import unittest

import logging
import os

from experiment_organizer.models import Session, Base
from experiment_organizer.models import ImageTypeEnum
from experiment_organizer.models import *
from experiment_organizer import ExperimentOrganizer


class ExperimentOrganizerTests(unittest.TestCase):
    experiment_path = os.path.join(
        '.', 'tests', 'test_data', 'users', 'testuser',
        '180823-UT-TestExperiment1'
    )
    experiment_organizer = None

    @classmethod
    def setUpClass(self):
        # initialize the ExperimentOrganizer as a global resource
        # for all unit tests
        self.experiment_organizer = ExperimentOrganizer(
            experiment_path=self.experiment_path,
            database_name=':memory:'
        ).__enter__()

    @classmethod
    def tearDownClass(self):
        # cleanup global resources
        self.experiment_organizer.__exit__(None, None, None)

    def test01_build_index(self):
        self.experiment_organizer.build_index(
            load_images=True,
            load_segmentations=True,
            load_segmentation_features=True,
            include_image_folders=[
                ('TIF', ImageTypeEnum.tif),
                ('TIF_OVR', ImageTypeEnum.overview),
                ('TIF_OVR_SEG', ImageTypeEnum.label)
            ],
            exclude_tags=['rgb_labels']
        )

        self.assertEqual(
            len(User.query.filter(User.name == 'testuser').all()), 1
        )

        self.assertEqual(
            len(Plate.query.all()), 2
        )

        self.assertEqual(
            len(Image.query.all()), 22
        )

        self.assertEqual(
            len(Segmentation.query.all()), 90
        )

        self.assertEqual(
            len(ImageChannelSegmentation.query.all()), 180
        )

        self.assertEqual(
            len(LocalImageChannelFeatureType.query.all()), 41
        )

        self.assertEqual(
            len(LocalImageChannelFeature.query.all()), 9090
        )

        self.experiment_organizer.commit()
        print(self.experiment_organizer._experiment)

    def test02_link_organoids(self):
        self.experiment_organizer.link_organoids()

        self.assertEqual(
            len(Object.query.all()), 45
        )

        fixation = Fixation.query.filter(Fixation.name == 'Day3full').one()
        segmentations = []
        for round, label in [(1, 250), (2, 250)]:
            segmentations.append(
                Segmentation.query
                .join(ImageChannelSegmentation).filter(
                    ImageChannelSegmentation.label == label
                )
                .join(ImageChannel)
                .join(Image)
                .join(Well).filter(Well.name == 'C3')
                .join(Plate).filter(
                    Plate.fixation_id == fixation.id
                ).filter(
                    Plate.round_id == round
                )
                .join(Experiment).one()
            )
        self.assertTrue(
            all(
                [
                    segmentations[0].object.id == segmentation.object.id
                    for segmentation in segmentations[1:]
                ]
            )
        )

        self.experiment_organizer.commit()


if __name__ == '__main__':
    unittest.main()
