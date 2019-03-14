#!/usr/bin/env python3

import os
import platform
import unittest

import numpy as np
from geoalchemy2.shape import from_shape, to_shape

import experiment_organizer
from experiment_organizer.models import init_sqlite_engine, Session, Base
from experiment_organizer.models import *
from experiment_organizer.utils import ATMatrix


class UtilsTests(unittest.TestCase):
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

    def test01_at_matrix_transform(self):
        session = Session()

        seg_matrix = ATMatrix(
            np.array([
                [1, 0, 0, 2],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        )
        img_matrix = ATMatrix(
            np.array([
                [1, 0, 0, 1],
                [0, 1, 0, 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        )

        image = Image(image_matrix=img_matrix.as_binary())
        image_channel = ImageChannel(image=image)
        image_segmentation = ImageChannelSegmentation(
            segmentation_matrix=seg_matrix.as_binary(),
            image_channel=image_channel,
            segmentation=Segmentation(
                contour=from_shape(
                    shapely.geometry.Polygon([(1, 1), (1, 1), (1, 1)])
                )
            )
        )

        session.add(image)
        session.add(image_segmentation)
        session.flush()

        # test Python properties
        self.assertEqual(
            to_shape(image_segmentation.contour_segmentation_space).wkt,
            "POLYGON ((1 1, 1 1, 1 1, 1 1))"
        )
        self.assertEqual(
            to_shape(image_segmentation.contour_image_space).wkt,
            "POLYGON ((3 2, 3 2, 3 2, 3 2))"
        )
        self.assertEqual(
            to_shape(image_segmentation.contour_microscope_space).wkt,
            "POLYGON ((4 4, 4 4, 4 4, 4 4))"
        )

        # test SQL properties
        self.assertEqual(
            shapely.wkt.loads(
                session.query(
                    func.AsText(
                        ImageChannelSegmentation.contour_segmentation_space
                    )
                ).filter(
                    ImageChannelSegmentation.id == image_segmentation.id
                ).first()[0]
            ).wkt,
            "POLYGON ((1 1, 1 1, 1 1, 1 1))"
        )
        self.assertEqual(
            shapely.wkt.loads(
                session.query(
                    func.AsText(ImageChannelSegmentation.contour_image_space)
                ).filter(
                    ImageChannelSegmentation.id == image_segmentation.id
                ).first()[0]
            ).wkt,
            "POLYGON ((3 2, 3 2, 3 2, 3 2))"
        )
        self.assertEqual(
            shapely.wkt.loads(
                session.query(
                    func.AsText(
                        ImageChannelSegmentation.contour_microscope_space
                    )
                ).filter(
                    ImageChannelSegmentation.id == image_segmentation.id
                ).first()[0]
            ).wkt,
            "POLYGON ((4 4, 4 4, 4 4, 4 4))"
        )

        session.close()


if __name__ == '__main__':
    unittest.main()
