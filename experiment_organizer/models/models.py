import datetime
import enum
import pandas as pd
import shapely.geometry
import shapely.affinity
import shapely.wkt

from sqlalchemy import Table, Sequence, Column, ForeignKey, UniqueConstraint
from sqlalchemy.types import \
    Integer, Float, String, DateTime, Time, Enum, LargeBinary, JSON
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func, select, join
from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape, to_shape

from experiment_organizer.models.base import Base
from experiment_organizer.utils import \
    DefaultDict, ATMatrix, init_empty_dataframe


class CoordinateSpaceEnum(enum.Enum):
    microscope = 0
    image = 1
    segmentation = 2


class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String)

    experiments = relationship("Experiment", back_populates="owner")

    __table_args__ = (
        UniqueConstraint('name', name='user_unique_idx'),
    )

    def __repr__(self):
        return "<User(name='%s')>" % (self.name,)


class ExperimentTypeEnum(enum.Enum):
    time_course = 1
    multiplexing = 2


class ExperimentType(Base):
    __tablename__ = 'experiment_type'

    id = Column(Integer, Sequence('experiment_type_id_seq'), primary_key=True)
    type = Column(Enum(ExperimentTypeEnum))
    experiment_id = Column(Integer, ForeignKey('experiment.id'))

    experiment = relationship("Experiment", back_populates="types")

    def __repr__(self):
        return "<ExperimentType(type='%s')>" % (self.type,)


class Experiment(Base):
    __tablename__ = 'experiment'

    id = Column(Integer, Sequence('experiment_id_seq'), primary_key=True)
    name = Column(String)
    path = Column(String)
    owner_id = Column(Integer, ForeignKey('user.id'))

    types = relationship("ExperimentType", back_populates="experiment")
    owner = relationship("User", back_populates="experiments")
    plates = relationship("Plate", back_populates="experiment")

    def __repr__(self):
        return (
            "<Experiment(name='%s', types='%s', path='%s', owner='%s')>"
            % (self.name, self.types, self.path, self.owner)
        )


class Fixation(Base):
    __tablename__ = 'fixation'

    id = Column(String, primary_key=True)
    name = Column(String)
    time_point = Column(Integer)

    plates = relationship("Plate", back_populates="fixation")

    __table_args__ = (
        UniqueConstraint('name', name='fixation_unique_idx'),
    )

    def __repr__(self):
        return (
            "<Fixation(id='%s', name='%s', time_point='%s')>"
            % (self.id, self.name, self.time_point)
        )


class Plate(Base):
    __tablename__ = 'plate'

    id = Column(Integer, Sequence('plate_id_seq'), primary_key=True)
    no = Column(Integer)
    round_id = Column(Integer)
    fixation_id = Column(String, ForeignKey('fixation.id'))
    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    bar_code = Column(String)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    experiment = relationship("Experiment", back_populates="plates")
    fixation = relationship("Fixation", back_populates="plates")
    wells = relationship("Well", back_populates="plate")

    __table_args__ = (
        UniqueConstraint(
            'no', 'round_id', 'fixation_id', 'experiment_id',
            name='plate_unique_idx'
        ),
    )

    def __repr__(self):
        return (
            (
                "<Plate(id='%s', round='%s', fixation='%s', bar_code='%s', "
                "created_at='%s')>"
            ) % (
                self.id, self.round_id, self.fixation, self.bar_code,
                self.created_at
            )
        )


class Well(Base):
    __tablename__ = 'well'

    def __init__(self, **kwargs):
        if 'id' not in kwargs:
            kwargs['name'] = (
                kwargs['row_id'] +
                str(kwargs['column_id'])[0:2].replace('0', '') +
                str(kwargs['column_id'])[2:]
            )
        super(Well, self).__init__(**kwargs)

    id = Column(Integer, Sequence('well_id_seq'), primary_key=True)
    plate_id = Column(Integer, ForeignKey('plate.id'))

    name = Column(
        String,
        default=(
            lambda context: (
                context.get_current_parameters()['row_id'] +
                str(context.get_current_parameters()['column_id']).zfill(2)
            )
        )
    )
    row_id = Column(String, nullable=False)
    column_id = Column(Integer, nullable=False)
    treatment = Column(String)

    plate = relationship("Plate", back_populates="wells")
    images = relationship("Image", back_populates="well")

    @property
    def stains(self):
        return (
            Stain.query
            .join(ImageChannel)
            .join(Image)
            .join(Well).filter(Well.id == self.id)
            .all()
        )

    __table_args__ = (
        UniqueConstraint('name', 'plate_id', name='well_unique_idx'),
    )

    def __repr__(self):
        return (
            "<Well(id='%s', plate='%s', treatment='%s')>"
            % (self.id, self.plate, self.treatment)
        )


class ImageTypeEnum(enum.Enum):
    tif = 1  # tif map
    overview = 2  # overview map
    mip = 3  # MIP map
    sip = 4  # SIP map
    rgb = 5  # RGB map
    label = 6  # label map


class Image(Base):
    __tablename__ = 'image'

    id = Column(Integer, Sequence('image_id_seq'), primary_key=True)
    name = Column(Integer)
    path = Column(Integer)
    type = Column(Enum(ImageTypeEnum), index=True)
    tag = Column(String)
    image_matrix = Column(LargeBinary, default=ATMatrix().as_binary())
    bbox = Column(Geometry(
        geometry_type='POLYGON', management=True, use_st_prefix=False
    ))

    time_point = Column(Integer)
    field_id = Column(Integer)
    time_line_id = Column(Integer)
    z_stack = Column(Integer)
    action_id = Column(Integer)

    well_id = Column(Integer, ForeignKey('well.id'))

    well = relationship("Well", back_populates="images")
    image_channels = relationship("ImageChannel", back_populates="image")

    def __repr__(self):
        return (
            "<Image(name='%s', path='%s', type='%s', tag='%s')>"
            % (self.name, self.path, self.type, self.tag)
        )


class ImageChannel(Base):
    __tablename__ = 'image_channel'

    id = Column(Integer, Sequence('image_channel_id_seq'), primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    channel_id = Column(Integer, ForeignKey('channel.id'))
    stain_id = Column(Integer, ForeignKey('stain.id'))

    image = relationship("Image", back_populates="image_channels")
    image_channel_segmentations = relationship(
        "ImageChannelSegmentation", back_populates="image_channel"
    )
    channel = relationship("Channel", back_populates="image_channels")
    stain = relationship("Stain", back_populates="image_channels")

    __table_args__ = (
        UniqueConstraint(
            'image_id', 'channel_id', 'stain_id',
            name='image_channel_unique_idx'
        ),
    )

    def __repr__(self):
        return (
            "<ImageChannel(image='%s', channel='%s', stain='%s')>"
            % (self.image, self.channel, self.stain)
        )


class Channel(Base):
    __tablename__ = 'channel'

    id = Column(Integer, Sequence('channel_id_seq'), primary_key=True)
    name = Column(String)
    wave_length = Column(Float)

    image_channels = relationship("ImageChannel", back_populates="channel")

    __table_args__ = (
        UniqueConstraint('name', name='channel_unique_idx'),
    )

    def __repr__(self):
        return "<Channel(name='%s', )>" % (self.name,)


class Stain(Base):
    __tablename__ = 'stain'

    id = Column(Integer, Sequence('stain_id_seq'), primary_key=True)
    name = Column(String)

    image_channels = relationship("ImageChannel", back_populates="stain")

    __table_args__ = (
        UniqueConstraint('name', name='stain_unique_idx'),
    )

    def __repr__(self):
        return "<Stain(name='%s')>" % (self.name,)


class ImageChannelSegmentation(Base):
    __tablename__ = 'image_channel_segmentation'

    id = Column(
        Integer, Sequence('image_channel_segmentation_id_seq'),
        primary_key=True
    )
    label = Column(Integer, index=True)
    segmentation_matrix = Column(LargeBinary, default=ATMatrix().as_binary())

    image_channel_id = Column(Integer, ForeignKey('image_channel.id'))
    segmentation_id = Column(Integer, ForeignKey('segmentation.id'))

    image_channel = relationship(
        "ImageChannel", back_populates="image_channel_segmentations"
    )
    local_image_channel_features = relationship(
        "LocalImageChannelFeature", back_populates="image_channel_segmentation"
    )
    segmentation = relationship(
        "Segmentation", back_populates="image_channel_segmentations"
    )

    def _projection(self, geometry, space=CoordinateSpaceEnum.segmentation):
        # @see: https://shapely.readthedocs.io/en/stable/
        #           manual.html#affine-transformations
        seg_matrix = ATMatrix(self.segmentation_matrix).as_3darray()
        shapely_seg_matrix = [
            seg_matrix[0, 0], seg_matrix[0, 1],
            seg_matrix[1, 0], seg_matrix[1, 1],
            seg_matrix[0, 3], seg_matrix[1, 3]
        ]
        img_matrix = ATMatrix(
            self.image_channel.image.image_matrix
        ).as_3darray()
        shapely_img_matrix = [
            img_matrix[0, 0], img_matrix[0, 1],
            img_matrix[1, 0], img_matrix[1, 1],
            img_matrix[0, 3], img_matrix[1, 3]
        ]

        if space in [
            CoordinateSpaceEnum.image, CoordinateSpaceEnum.microscope
        ]:
            geometry = shapely.affinity.affine_transform(
                geometry,
                shapely_seg_matrix
            )
        if space == CoordinateSpaceEnum.microscope:
            geometry = shapely.affinity.affine_transform(
                geometry,
                shapely_img_matrix
            )

        return geometry

    @hybrid_property
    def bbox_segmentation_space(self):
        # usage: ImageChannelSegmentation(...).bbox_segmentation_space
        return from_shape(
            self._projection(
                to_shape(self.segmentation.contour).envelope,
                CoordinateSpaceEnum.segmentation
            )
        )

    @hybrid_property
    def bbox_image_space(self):
        return from_shape(
            self._projection(
                to_shape(self.segmentation.contour).envelope,
                CoordinateSpaceEnum.image
            )
        )

    @hybrid_property
    def bbox_microscope_space(self):
        return from_shape(
            self._projection(
                to_shape(self.segmentation.contour).envelope,
                CoordinateSpaceEnum.microscope
            )
        )

    @bbox_segmentation_space.expression
    def bbox_segmentation_space(cls):
        # usage:
        # session.query(ImageChannelSegmentation.bbox_segmentation_space)
        return (
            select([
                func.ATM_Transform(
                    func.Envelope(Segmentation.contour), func.ATM_Create()
                )
            ])
            .where(Segmentation.id == cls.segmentation_id)
            .correlate(ImageChannelSegmentation)
            .label('bbox_segmentation_space')
        )

    @bbox_image_space.expression
    def bbox_image_space(cls):
        return (
            select([
                func.ATM_Transform(
                    func.Envelope(Segmentation.contour),
                    cls.segmentation_matrix
                )
            ])
            .where(Segmentation.id == cls.segmentation_id)
            .correlate(ImageChannelSegmentation)
            .label('bbox_image_space')
        )

    @bbox_microscope_space.expression
    def bbox_microscope_space(cls):
        return (
            select([
                func.ATM_Transform(
                    func.ATM_Transform(
                        func.Envelope(Segmentation.contour),
                        cls.segmentation_matrix
                    ),
                    Image.image_matrix
                )
            ])
            .where(
                (Segmentation.id == cls.segmentation_id) &
                (ImageChannel.id == cls.image_channel_id) &
                (Image.id == ImageChannel.image_id)
            )
            .correlate(ImageChannelSegmentation)
            .label('bbox_microscope_space')
        )

    @hybrid_property
    def contour_segmentation_space(self):
        return from_shape(
            self._projection(
                to_shape(self.segmentation.contour),
                CoordinateSpaceEnum.segmentation
            )
        )

    @hybrid_property
    def contour_image_space(self):
        return from_shape(
            self._projection(
                to_shape(self.segmentation.contour),
                CoordinateSpaceEnum.image
            )
        )

    @hybrid_property
    def contour_microscope_space(self):
        return from_shape(
            self._projection(
                to_shape(self.segmentation.contour),
                CoordinateSpaceEnum.microscope
            )
        )

    @contour_segmentation_space.expression
    def contour_segmentation_space(cls):
        return (
            select([
                func.ATM_Transform(
                    Segmentation.contour, func.ATM_Create()
                )
            ])
            .where(Segmentation.id == cls.segmentation_id)
            .correlate(ImageChannelSegmentation)
            .label('contour_segmentation_space')
        )

    @contour_image_space.expression
    def contour_image_space(cls):
        return (
            select([
                func.ATM_Transform(
                    Segmentation.contour, cls.segmentation_matrix
                )
            ])
            .where(Segmentation.id == cls.segmentation_id)
            .correlate(ImageChannelSegmentation)
            .label('contour_image_space')
        )

    @contour_microscope_space.expression
    def contour_microscope_space(cls):
        return (
            select([
                func.ATM_Transform(
                    func.ATM_Transform(
                        Segmentation.contour, cls.segmentation_matrix
                    ),
                    Image.image_matrix
                )
            ])
            .where(
                (Segmentation.id == cls.segmentation_id) &
                (ImageChannel.id == cls.image_channel_id) &
                (Image.id == ImageChannel.image_id)
            )
            .correlate(ImageChannelSegmentation)
            .label('contour_microscope_space')
        )

    def __repr__(self):
        return (
            "<ImageChannelSegmentation(image='%s', channel='%s', label='%s')>"
            % (
                self.image_channel.image,
                self.image_channel.channel,
                self.label
            )
        )


class LocalImageChannelFeatureType(Base):
    __tablename__ = 'local_image_channel_feature_type'

    id = Column(
        Integer, Sequence('local_image_channel_feature_type_id_seq'),
        primary_key=True
    )
    name = Column(String)
    parent_id = Column(
        Integer, ForeignKey('local_image_channel_feature_type.id')
    )

    local_image_channel_features = relationship(
        "LocalImageChannelFeature",
        back_populates="local_image_channel_feature_type"
    )
    children = relationship(
        "LocalImageChannelFeatureType",
        backref=backref('parent', remote_side=[id])
    )

    __table_args__ = (
        UniqueConstraint(
            'parent_id', 'name',
            name='local_image_channel_feature_type_unique_idx'
        ),
    )

    def __repr__(self):
        return (
            "<LocalImageChannelFeatureType(parent='%s', name='%s')>"
            % (self.parent, self.name)
        )


class LocalImageChannelFeature(Base):
    __tablename__ = 'local_image_channel_feature'

    id = Column(
        Integer, Sequence('local_image_channel_feature_id_seq'),
        primary_key=True
    )
    value = Column(String)

    image_channel_segmentation_id = Column(
        Integer, ForeignKey('image_channel_segmentation.id')
    )
    local_image_channel_feature_type_id = Column(
        Integer, ForeignKey('local_image_channel_feature_type.id')
    )

    image_channel_segmentation = relationship(
        "ImageChannelSegmentation",
        back_populates="local_image_channel_features"
    )
    local_image_channel_feature_type = relationship(
        "LocalImageChannelFeatureType",
        back_populates="local_image_channel_features"
    )

    def __repr__(self):
        return (
            "<LocalImageChannelFeature(%s='%s')>"
            % (self.local_image_channel_feature_type, self.value)
        )


class Segmentation(Base):
    __tablename__ = 'segmentation'

    id = Column(Integer, Sequence('segmentation_id_seq'), primary_key=True)
    object_id = Column(Integer, ForeignKey('object.id'))

    # contour is stored as WKBElement
    contour = Column(Geometry(
        geometry_type='POLYGON', management=True, use_st_prefix=False
    ))

    image_channel_segmentations = relationship(
        "ImageChannelSegmentation", back_populates="segmentation"
    )
    shape_features = relationship(
        "ShapeFeature", back_populates="segmentation"
    )
    object = relationship("Object", back_populates="segmentations")

    def __repr__(self):
        return (
            "<ImageSegmentation(id='%s', contour='%s')>"
            % (self.id, to_shape(self.contour).wkt)
        )


class ShapeFeatureType(Base):
    __tablename__ = 'shape_feature_type'

    id = Column(
        Integer, Sequence('shape_feature_type_id_seq'), primary_key=True
    )
    name = Column(String)
    parent_id = Column(Integer, ForeignKey('shape_feature_type.id'))

    shape_features = relationship(
        "ShapeFeature", back_populates="shape_feature_type"
    )
    children = relationship(
        "ShapeFeatureType", backref=backref('parent', remote_side=[id])
    )

    __table_args__ = (
        UniqueConstraint(
            'parent_id', 'name', name='shape_feature_type_unique_idx'
        ),
    )

    def __repr__(self):
        return (
            "<ShapeFeatureType(parent='%s', name='%s')>"
            % (self.parent, self.name)
        )


class ShapeFeature(Base):
    __tablename__ = 'shape_feature'

    id = Column(Integer, Sequence('shape_feature_id_seq'), primary_key=True)
    value = Column(String)

    segmentation_id = Column(Integer, ForeignKey('segmentation.id'))
    shape_feature_type_id = Column(
        Integer, ForeignKey('shape_feature_type.id')
    )

    segmentation = relationship(
        "Segmentation", back_populates="shape_features"
    )
    shape_feature_type = relationship(
        "ShapeFeatureType", back_populates="shape_features"
    )

    def __repr__(self):
        return (
            "<ShapeFeature(%s='%s')>" % (self.shape_feature_type, self.value)
        )


class ObjectTypeEnum(enum.Enum):
    organoid = 1
    cell = 2
    nucleus = 3


class Object(Base):
    __tablename__ = 'object'

    id = Column(Integer, Sequence('object_id_seq'), primary_key=True)
    type = Column(Enum(ObjectTypeEnum))

    segmentations = relationship("Segmentation", back_populates="object")

    def __repr__(self):
        return "<Object(type='%s')>" % (self.type,)


class Organoid(Base):
    __tablename__ = 'organoid'

    id = Column(Integer, ForeignKey('object.id'), primary_key=True)

    object = relationship("Object")
    cells = relationship("Cell", back_populates="organoid")

    def __repr__(self):
        return "<Organoid(id='%s')>" % (self.id,)


class Cell(Base):
    __tablename__ = 'cell'

    id = Column(Integer, ForeignKey('object.id'), primary_key=True)
    organoid_id = Column(Integer, ForeignKey('organoid.id'))

    object = relationship("Object")
    organoid = relationship("Organoid", back_populates="cells")

    def __repr__(self):
        return "<Cell(id='%s', organoid='%s')>" % (self.id, self.organoid)


# non db models
class MeasurementDataFrame(pd.DataFrame):
    _metadata = ["file_path"]

    @property
    def _constructor(self):
        return self.__class__

    def __init__(self, data=None, index=None, file_path=None):
        expected_columns = [
            'type', 'time',
            'well_name', 'column', 'row', 'time_point', 'field_id',
            'partial_tile_id', 'tile_x_id', 'tile_y_id', 'z_index',
            'timeline_id', 'action_id', 'action',
            'x_micrometer', 'y_micrometer', 'z_micrometer',
            'x_pixel', 'y_pixel', 'bit_depth', 'width', 'height',
            'channel_id', 'camera_no',
            'file_path', 'file_name'
        ]

        if data is not None:
            data = pd.DataFrame(data)
            if not all([
                column for column in expected_columns if column in data.columns
            ]):
                raise RuntimeError(
                    "Cannot initialize MeasurementDataFrame with the given "
                    "DataFrame due missing columns!"
                )
        else:
            data = init_empty_dataframe(
                columns=[
                    'type', 'time',
                    'well_name', 'column', 'row', 'time_point', 'field_id',
                    'partial_tile_id', 'tile_x_id', 'tile_y_id', 'z_index',
                    'timeline_id', 'action_id', 'action',
                    'x_micrometer', 'y_micrometer', 'z_micrometer',
                    'x_pixel', 'y_pixel', 'bit_depth', 'width', 'height',
                    'channel_id', 'camera_no',
                    'file_path', 'file_name'
                ],
                dtypes=[
                    'str', 'datetime64[ns]',
                    'str', 'int', 'int', 'int', 'int',
                    'int', 'int', 'int', 'int',
                    'int', 'int', 'str',
                    'float', 'float', 'float',
                    'int', 'int', 'int', 'int', 'int',
                    'int', 'int',
                    'str', 'str'
                ],
                index=index
            )

        # call parent constructor
        super(MeasurementDataFrame, self).__init__(
            data=data,
            index=index
        )

        # additional attributes
        self.file_path = file_path

    def get_bounding_boxes_per_well(self):
        well_groups = self.groupby('well_name')

        well_bbox_df = pd.DataFrame()
        well_bbox_df = pd.DataFrame(
            columns=["bbox"], index=list(well_groups.groups)
        )

        # calculate bounding-box
        for well_name, well_group in well_groups:
            # upper-left pixel coordinates in microscope space
            ul_x = well_group['x_pixel'].min()
            ul_y = well_group['y_pixel'].min()

            # lower-right pixel coordinates in microscope space
            lr_x = well_group['x_pixel'].max() + well_group['width'].iloc[0]
            lr_y = (
                well_group['y_pixel'].max() + well_group['height'].iloc[0] + 1
            )

            well_bbox_df.at[well_name, 'bbox'] = shapely.geometry.box(
                ul_x, ul_y, lr_x, lr_y
            )

        return well_bbox_df


class MeasurementDetailFrame(pd.DataFrame):
    _metadata = ["file_path"]

    @property
    def _constructor(self):
        return self.__class__

    def __init__(self, data=None, index=None, file_path=None):
        expected_columns = [
            'channel_id',
            'horizontal_pixel_dim', 'vertical_pixel_dim', 'camera_no',
            'input_bit_depth', 'input_level',
            'horizontal_pixels', 'vertical_pixels',
            'filter_wheel_pos', 'filter_pos', 'shading_corr_src'
        ]

        if data is not None:
            data = pd.DataFrame(data)
            if not all([
                column for column in expected_columns if column in data.columns
            ]):
                raise RuntimeError(
                    "Cannot initialize MeasurementDetailFrame with the given "
                    "DataFrame due missing columns!"
                )
        else:
            data = init_empty_dataframe(
                columns=[
                    'channel_id',
                    'horizontal_pixel_dim', 'vertical_pixel_dim', 'camera_no',
                    'input_bit_depth', 'input_level',
                    'horizontal_pixels', 'vertical_pixels',
                    'filter_wheel_pos', 'filter_pos', 'shading_corr_src'
                ],
                dtypes=[
                    'int',
                    'float', 'float', 'int',
                    'int', 'int',
                    'int', 'int',
                    'int', 'int', 'str'
                ],
                index=index
            )

        # call parent constructor
        super(MeasurementDetailFrame, self).__init__(
            data=data,
            index=index
        )

        # additional attributes
        self.file_path = file_path


class ExperimentLayout():
    _plate_condition_map = {}
    _plate_stain_map = {}

    def __init__(self, plate_condition_map={}, plate_stain_map={}):
        self._plate_condition_map = DefaultDict(plate_condition_map)
        self._plate_stain_map = DefaultDict(plate_stain_map)

    @property
    def plate_condition_map(self):
        return self._plate_condition_map

    @plate_condition_map.setter
    def plate_condition_map(self, plate_condition_map):
        self._plate_condition_map = DefaultDict(plate_condition_map)

    @property
    def plate_stain_map(self):
        return self._plate_stain_map

    @plate_stain_map.setter
    def plate_stain_map(self, plate_stain_map):
        self._plate_stain_map = DefaultDict(plate_stain_map)


class SegmentationFeatureFrame(pd.DataFrame):
    _metadata = ["file_path", "feature_classes"]

    @property
    def _constructor(self):
        return self.__class__

    def __init__(self, data=None, index=None, file_path=None):
        expected_columns = [
            'well_name', 'object_number'
        ]

        if data is not None:
            data = pd.DataFrame(data)
            if not all([
                column for column in expected_columns if column in data.columns
            ]):
                raise RuntimeError(
                    "Cannot initialize SegmentationFeatureFrame with the "
                    "given DataFrame due missing columns!"
                )
        else:
            data = init_empty_dataframe(
                columns=['well_name', 'object_number'],
                dtypes=['str', 'int'],
                index=index
            )

        # call parent constructor
        super(SegmentationFeatureFrame, self).__init__(
            data=data,
            index=index
        )

        # additional attributes
        self.file_path = file_path
        self.feature_classes = []
