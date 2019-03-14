#!/usr/bin/env python3

import os
import platform
import unittest

import numpy as np
from shapely.geometry import box

from geoalchemy2.shape import from_shape, to_shape

import experiment_organizer
from experiment_organizer.models import init_sqlite_engine, Session, Base
from experiment_organizer.models import *
from experiment_organizer.utils import ATMatrix


class ModelsTests(unittest.TestCase):
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

    def test01_image_scaling(self):
        session = Session()

        # <all coordinates in microscope space>
        #
        # +(0,0)-------------------------------  microscope space
        # |
        # | +(1,1)-----------------------------  image (scaling = 1:2)
        # | |
        # | |
        # | |
        # | |   +(3,3)  segmentation
        # | |   |   |
        # | |   |   |
        # | |   |   |
        # | |   +---+(5,5)
        # | |

        # using SQLAlchemy models
        # ---------------------------------------------------------------------
        origin_microscope_space = (1, 1)
        shrinkage_level = 2
        image_translation_matrix = np.array([
            [1, 0, 0, origin_microscope_space[0]/shrinkage_level],
            [0, 1, 0, origin_microscope_space[1]/shrinkage_level],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.double)
        image_scaling_matrix = np.array([
            [shrinkage_level, 0, 0, 0],
            [0, shrinkage_level, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.double)

        # define transformation from image to microscope space
        image_matrix = ATMatrix(np.matmul(
            image_scaling_matrix, image_translation_matrix
        ))

        image = Image(
            image_matrix=image_matrix.as_binary()
        )

        origin_image_space = (1, 1)
        seg_translation_matrix = np.array([
            [1, 0, 0, origin_image_space[0]],
            [0, 1, 0, origin_image_space[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.double)

        # define transformation from segmentation to image space
        segegmentation_matrix = ATMatrix(seg_translation_matrix)

        image_cn_seg = ImageChannelSegmentation(
            image_channel=ImageChannel(
                image=image,
                channel=None
            ),
            segmentation=Segmentation(
                contour=from_shape(shapely.geometry.box(0.0, 0.0, 1.0, 1.0))
            ),
            segmentation_matrix=segegmentation_matrix.as_binary()
        )

        # looking from the segmentation space
        self.assertEqual(
            to_shape(image_cn_seg.contour_segmentation_space).bounds,
            (0.0, 0.0, 1.0, 1.0)
        )
        # looking from the image space
        self.assertEqual(
            to_shape(image_cn_seg.contour_image_space).bounds,
            (1.0, 1.0, 2.0, 2.0)
        )
        # looking from the microscope space
        # (which takes the shrinkage level of the image into account)
        self.assertEqual(
            to_shape(image_cn_seg.contour_microscope_space).bounds,
            (3.0, 3.0, 5.0, 5.0)
        )

        # using SQL
        # ---------------------------------------------------------------------
        res = session.query(
            func.AsText(
                func.ATM_Transform(
                    func.ST_GeomFromText('POINT(0 0)'),
                    func.ATM_Scale(
                        func.ATM_Translate(
                            func.ATM_CreateTranslate(
                                origin_image_space[0], origin_image_space[1]
                            ),
                            origin_microscope_space[0]/shrinkage_level,
                            origin_microscope_space[1]/shrinkage_level
                        ),
                        shrinkage_level, shrinkage_level
                    )
                )
            ).label('pt')
        ).one()
        pt_microscope_space = shapely.wkt.loads(res.pt)

        self.assertEqual(
            pt_microscope_space.coords[0], (3, 3)
        )

        # reverse transform image point into microscope space
        res = session.query(
            func.AsText(
                func.ATM_Transform(
                    func.ST_GeomFromText('POINT(3 3)'),
                    func.ATM_Translate(
                        func.ATM_Translate(
                            func.ATM_CreateScale(
                                1/shrinkage_level, 1/shrinkage_level
                            ),
                            -origin_microscope_space[0]/shrinkage_level,
                            -origin_microscope_space[1]/shrinkage_level
                        ),
                        -origin_image_space[0], -origin_image_space[1]
                    )
                )
            ).label('pt')
        ).one()
        pt_segmentation_space = shapely.wkt.loads(res.pt)
        self.assertEqual(
            pt_segmentation_space.coords[0], (0, 0)
        )

        session.close()


if __name__ == '__main__':
    unittest.main()
