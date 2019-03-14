#!/usr/bin/env python3

import os
import platform
import unittest

from geoalchemy2.shape import from_shape, to_shape

import experiment_organizer
from experiment_organizer.models import init_sqlite_engine, Session, Base
from experiment_organizer.models import *
from experiment_organizer.workflows import OrganoidLinkingWorkflow


class WorkflowsTests(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(self):
        if platform.system() == 'Windows':
            mod_spatialite_path = os.path.join(
                os.path.dirname(experiment_organizer.__file__),
                'mod_spatialite-wn'
            )
        elif platform.system() == 'Linux':
            mod_spatialite_path = os.path.join(
                os.path.dirname(experiment_organizer.__file__),
                'mod_spatialite-lx'
            )
        else:
            mod_spatialite_path = os.path.join(
                os.path.dirname(experiment_organizer.__file__),
                'mod_spatialite'
            )

        self.engine = init_sqlite_engine(
            ':memory:',
            mod_spatialite_path=mod_spatialite_path
        )
        Base.metadata.create_all(self.engine)  # generate database schema

    @classmethod
    def tearDownClass(self):
        self.engine.dispose()

    def test01_organoid_linking(self):
        session = Session()

        experiment = Experiment()

        fixation = Fixation(id="f1", name="Day1full", time_point=1)
        channel = Channel(name='c1')
        for round_id in range(1, 10):
            plate = Plate(
                round_id=round_id, fixation=fixation, experiment=experiment
            )
            well = Well(row_id="A", column_id=1, plate=plate)
            image = Image(type=ImageTypeEnum.label, tag="1", well=well)
            if round_id == 1:
                ImageChannelSegmentation(
                    label=1,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(0, 0, 1000, 1000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
                ImageChannelSegmentation(
                    label=2,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(2000, 0, 3000, 1000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
            if round_id == 2:
                ImageChannelSegmentation(
                    label=3,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(250, 250, 1250, 1250)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
            if round_id == 3:
                pass
            if round_id == 4:
                pass
            if round_id == 5:
                ImageChannelSegmentation(
                    label=4,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(5000, 0, 6000, 1000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
            if round_id == 6:
                ImageChannelSegmentation(
                    label=5,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(0, 0, 1000, 1000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
                ImageChannelSegmentation(
                    label=6,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(5000, 0, 7000, 2000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
            if round_id == 7:
                ImageChannelSegmentation(
                    label=7,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(0, 0, 1000, 1000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
                ImageChannelSegmentation(
                    label=8,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(5000, 0, 6000, 1000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )
            if round_id == 8:
                ImageChannelSegmentation(
                    label=9,
                    segmentation=Segmentation(
                        contour=from_shape(
                            shapely.geometry.box(0, 0, 1000, 1000)
                        )
                    ),
                    image_channel=ImageChannel(image=image, channel=channel)
                )

        experiment.add()

        OrganoidLinkingWorkflow().run(experiment)

        self.assertEqual(
            len(Object.query.all()), 7
        )

        fixation = Fixation.query.filter(Fixation.name == 'Day1full').one()
        segmentations = []
        for round, label in [(6, 5), (7, 7), (8, 9)]:
            segmentations.append(
                Segmentation.query
                .join(ImageChannelSegmentation).filter(
                    ImageChannelSegmentation.label == label
                )
                .join(ImageChannel)
                .join(Image)
                .join(Well).filter(Well.name == 'A1')
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

        session.close()


if __name__ == '__main__':
    unittest.main()
