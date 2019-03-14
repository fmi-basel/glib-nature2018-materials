from sqlalchemy.ext import compiler
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql import select, join, table

from experiment_organizer.models import *


class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class DropView(DDLElement):
    def __init__(self, name):
        self.name = name


@compiler.compiles(CreateView)
def compile(element, compiler, **kw):
    return 'CREATE VIEW IF NOT EXISTS %s AS %s' % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True)
    )


@compiler.compiles(DropView)
def compile(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)


def view(name, metadata, selectable):
    t = table(name)

    for c in selectable.c:
        c._make_proxy(t)

    CreateView(name, selectable).execute_at('after-create', metadata)
    DropView(name).execute_at('before-drop', metadata)
    return t


class WellView(Base):
    __table__ = view(
        "well_view",
        Base.metadata,
        select(  # join: produce an INNER JOIN between left and right clauses
            [
                Experiment.id, Experiment.name,
                Plate.no, Plate.round_id, Fixation.name, Fixation.time_point,
                Well.name, Well.treatment
            ], use_labels=True
        ).select_from(
            join(Experiment, Plate, Experiment.id == Plate.experiment_id) \
            .join(Fixation, Plate.fixation_id == Fixation.id) \
            .join(Well, Plate.id == Well.plate_id)
        )
    )


class ImageChannelSegmentationView(Base):
    __table__ = view(
        "image_channel_segmentation_view",
        Base.metadata,
        select(  # join: produce an INNER JOIN between left and right clauses
            [
                Experiment.id, Experiment.name,
                Plate.no, Plate.round_id, Fixation.name, Fixation.time_point,
                Well.name, Well.treatment,
                Image.name, Image.type, Image.tag, Image.z_stack,
                Channel.name,
                ImageChannelSegmentation.id, ImageChannelSegmentation.label,
                ImageChannelSegmentation.contour_segmentation_space.label(
                    'image_channel_segmentation_contour_segmentation_space'
                ),
                ImageChannelSegmentation.contour_microscope_space.label(
                    'image_channel_segmentation_contour_microscope_space'
                )
                # , LocalImageChannelFeatureType.name
                # , LocalImageChannelFeature.value
            ], use_labels=True
        ).select_from(
            join(Experiment, Plate, Experiment.id == Plate.experiment_id)
            .join(Fixation, Plate.fixation_id == Fixation.id)
            .join(Well, Plate.id == Well.plate_id)
            .join(Image, Well.id == Image.well_id)
            .join(ImageChannel, Image.id == ImageChannel.image_id)
            .join(Channel, ImageChannel.channel_id == Channel.id)
            .join(
                ImageChannelSegmentation,
                ImageChannel.id == ImageChannelSegmentation.image_channel_id
            )
            # .join(
            #     LocalImageChannelFeature,
            #     ImageChannelSegmentation.id ==
            #     LocalImageChannelFeature.image_channel_segmentation_id
            # )
            # .join(
            #     LocalImageChannelFeatureType,
            #     LocalImageChannelFeatureType.id ==
            #     LocalImageChannelFeature.local_image_channel_feature_type_id
            # )
        )
    )


class ObjectView(Base):
    __table__ = view(
        "object_view",
        Base.metadata,
        select(  # join: produce an INNER JOIN between left and right clauses
            [
                Experiment.id, Experiment.name,
                Plate.no, Plate.round_id, Fixation.name, Fixation.time_point,
                Well.name, Well.treatment,
                Image.name, Image.type, Image.tag, Image.z_stack,
                Channel.name,
                ImageChannelSegmentation.id, ImageChannelSegmentation.label,
                ImageChannelSegmentation.contour_segmentation_space.label(
                    'image_channel_segmentation_contour_segmentation_space'
                ),
                ImageChannelSegmentation.contour_microscope_space.label(
                    'image_channel_segmentation_contour_microscope_space'
                ),
                Segmentation.id,
                # LocalImageChannelFeatureType.name,
                # LocalImageChannelFeature.value,
                Object.id, Object.type
            ], use_labels=True
        ).select_from(
            join(Experiment, Plate, Experiment.id == Plate.experiment_id)
            .join(Fixation, Plate.fixation_id == Fixation.id)
            .join(Well, Plate.id == Well.plate_id)
            .join(Image, Well.id == Image.well_id)
            .join(ImageChannel, Image.id == ImageChannel.image_id)
            .join(Channel, ImageChannel.channel_id == Channel.id)
            .join(
                ImageChannelSegmentation,
                ImageChannel.id == ImageChannelSegmentation.image_channel_id
            )
            # .join(
            #     LocalImageChannelFeature,
            #     ImageSegmentation.id ==
            #     LocalImageChannelFeature.image_segmentation_id
            # )
            # .join(
            #     LocalImageChannelFeatureType,
            #     LocalImageChannelFeatureType.id ==
            #     LocalImageChannelFeature.local_image_feature_type_id
            # )
            .join(
                Segmentation,
                Segmentation.id == ImageChannelSegmentation.segmentation_id
            )
            .join(Object, Object.id == Segmentation.object_id)
        )
    )
