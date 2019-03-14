import datetime
import os
import re
import imghdr
import numpy as np
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import box, Polygon
from skimage.io import imread
from skimage.util import pad
from skimage.measure import label, find_contours, regionprops
from PIL import Image as PILImage

from experiment_organizer.models import *
from experiment_organizer.readers.file_readers import \
    ExperimentMetaDataReader, ExperimentLayoutReader, \
    MeasurementDataReader, SegmentationFeatureReader
from experiment_organizer.utils import get_sub_paths, ATMatrix


class ExperimentReader(object):
    @staticmethod
    def read(
        experiment_path,
        load_images=True, load_segmentations=True,
        load_segmentation_features=True,
        include_image_folders=[], exclude_tags=[]
    ):
        experiment_name = os.path.basename(os.path.abspath(experiment_path))
        experiment_path_separation = experiment_path.lower().split(os.path.sep)
        if "users" in experiment_path_separation:
            experiment_owner = experiment_path_separation[
                experiment_path_separation.index('users')+1
            ]
        else:
            experiment_owner = (
                input("Please enter the experiment owner: ") or None
            )

        users = User.query.filter(User.name == experiment_owner).all()
        if len(users) == 1:
            user = users[0]
        else:
            user = User(name=experiment_owner)
        experiment = Experiment(
            name=experiment_name, path=experiment_path, owner=user
        )

        expr_path_level = os.path.abspath(experiment.path).count(os.path.sep)
        rounds = set()
        fixations = set()
        for root, dirs, files in os.walk(experiment.path, topdown=True):
            for dir in dirs:
                match = re.search(
                    r"^(\d{6})(\w{2})(\d)(.{2})$", os.path.basename(dir)
                )
                if match:
                    (
                        creation_date, owner, round_id, fixation_id
                    ) = match.groups()
                    rounds.add(round_id)
                    fixations.add(fixation_id)
            if expr_path_level + 2 <= os.path.abspath(root).count(os.path.sep):
                # stop traversing after reaching 3rd sub-directory level
                del dirs[:]

        if len(rounds) > 1:
            experiment.types.append(
                ExperimentType(type=ExperimentTypeEnum.multiplexing)
            )
        if len(fixations) > 1:
            experiment.types.append(
                ExperimentType(type=ExperimentTypeEnum.time_course)
            )

        ExperimentReader.read_plate_information(experiment)
        # flush experiment and all dependencies into the database
        experiment.add()

        if load_images:
            ExperimentReader.read_image_information(
                experiment, include_image_folders, exclude_tags
            )
            # flush experiment and all dependencies into the database
            experiment.add()
        if load_segmentations:
            ExperimentReader.read_segmentation_information(experiment)
            # flush experiment and all dependencies into the database
            experiment.add()
        if load_segmentation_features:
            ExperimentReader.read_segmentation_features(experiment)
            # flush experiment and all dependencies into the database
            experiment.add()

        return experiment

    @staticmethod
    def read_plate_information(experiment):
        meta_data = ExperimentMetaDataReader.read(
            os.path.join(experiment.path, 'meta.xlsx')
        )
        if len(experiment.plates) > 0:
            print(
                (
                    "WARNING: found existing plates for experiment '%s'! "
                    "Existing plates are replaced."
                ) % experiment
            )
            session.query(experiment.plates).delete()

        for plate in ExperimentReader._read_plate_information(
            experiment.path, meta_data
        ):
            experiment.plates.append(plate)

    @staticmethod
    def _read_plate_information(folder_path, meta_data, tags={}):
        for sub_folder_path in get_sub_paths(folder_path, dir_only=True):
            sub_folder_name = os.path.basename(
                os.path.abspath(sub_folder_path)
            )
            match = re.search(r"^(\d{6})(\w{2})(\d)(.{2})$", sub_folder_name)
            if match:
                # TODO: we miss the plate_id here (or the fixation_id?) ...
                creation_date, owner, round_id, fixation_id = match.groups()

                plate_no = tags['plate_no'] if 'plate_no' in tags else 1

                try:
                    creation_date = datetime.datetime.strptime(
                        creation_date, '%y%m%d'
                    )
                except ValueError:
                    print(
                        (
                            "WARNING: Bar code's creation date of plate "
                            "'%s' is in a non-standardized format!"
                        ) % (sub_folder_path,)
                    )
                    creation_date = datetime.datetime.strptime(
                        creation_date, '%y%d%m'
                    )
                if creation_date.year > datetime.datetime.now().year:
                    # check for date overflow
                    creation_date = creation_date.replace(
                        year=creation_date.year-100
                    )

                fixations = \
                    Fixation.query.filter(Fixation.id == fixation_id).all()
                if len(fixations) == 1:
                    fixation = fixations[0]
                else:
                    if fixation_id in meta_data['fixation'].index:
                        fixation_name = \
                            meta_data['fixation'].loc[fixation_id, 'name']
                        fixation = Fixation(
                            id=fixation_id,
                            name=fixation_name,
                            time_point=int(
                                meta_data['fixation']
                                .loc[fixation_id, 'lapse_sec']
                            )
                        )
                        # without adding the feature imediately to the db we
                        # would not find it in a later recursion while checking
                        # for its existence
                        fixation.add()
                    else:
                        print(
                            "WARNING: No meta data found for fixation id '%s'!"
                            % (fixation_id,)
                        )
                        # TODO: should we set fixation = None?
                        fixation = Fixation(
                            id=fixation_id, name=None, time_point=None
                        )
                        # without adding the feature imediately to the db we
                        # would not find it in a later recursion while checking
                        # for its existence
                        fixation.add()

                # TODO: How to decode the plate number?
                yield Plate(
                    no=plate_no,
                    round_id=round_id,
                    fixation=fixation,
                    bar_code=sub_folder_name,
                    created_at=creation_date,
                    # path=os.path.abspath(os.path.join(folder_path, '..'))
                    path=os.path.abspath(sub_folder_path)
                )
            else:
                match = re.search(
                    r"^(?:"
                    r"([Pp]late\s?\d+)|([Rr]ound\s?\d+)|([Ff]ixation\s?\d+)"
                    r")$",
                    sub_folder_name
                )
                if match is None:
                    print(
                        (
                            "WARNING: found unexpected tag '%s' in %s! "
                            "Folders within the tag folder are ignored."
                        ) % (sub_folder_name, os.path.abspath(sub_folder_path))
                    )
                else:
                    plate_no, round_id, fixation_id = match.groups()
                    tags = tags.copy()
                    if plate_no is not None:
                        tags["plate_no"] = plate_no
                    if round_id is not None:
                        print(
                            "INFO: a 'ROUND' tag was found which will be "
                            "overwritten by the bar code's round ID!"
                        )
                        tags["round_id"] = round_id
                    if fixation_id is not None:
                        print(
                            "INFO: a 'FIXATION' tag was found which will be "
                            "overwritten by the bar code's fixation ID!"
                        )
                        tags["fixation_id"] = fixation_id
                    yield from ExperimentReader._read_plate_information(
                        sub_folder_path, meta_data, tags
                    )

    @staticmethod
    def read_image_information(
        experiment, include_image_folders=[], exclude_tags=[]
    ):
        experiment_layout = ExperimentLayoutReader.read(
            os.path.join(experiment.path, 'layout.xlsx')
        )

        for plates_processed, plate in enumerate(experiment.plates):
            print(
                "Processing plate %s/%s"
                % (plates_processed+1, len(experiment.plates))
            )

            if len(plate.wells) > 0:
                print(
                    (
                        "WARNING: found existing wells for plate '%s'! "
                        "Existing wells are replaced."
                    ) % plate
                )
                session.query(plate.wells).delete()

            measurement_data_path = os.path.join(
                plate.path, "META", "MeasurementData.mlf"
            )

            try:
                mlf_frame = MeasurementDataReader.read(measurement_data_path)
            except RuntimeError as ex:
                print(
                    "WARNING: Skip plate '%s'!\n%s" % (plate, str(ex))
                )
                continue

            well_bbox_df = mlf_frame.get_bounding_boxes_per_well()

            condition_df = experiment_layout.plate_condition_map.get(
                plate.bar_code, try_default=True
            )
            stain_df = experiment_layout.plate_stain_map.get(
                plate.bar_code, try_default=True
            )

            channels = {
                channel.id: channel for channel in Channel.query.all()
            }
            wells = {}

            for image_folder_path in get_sub_paths(plate.path, dir_only=True):
                image_folder_name = os.path.basename(
                    os.path.abspath(image_folder_path)
                )

                image_type = None
                for valid_folder_name, type in include_image_folders:
                    if valid_folder_name == image_folder_name:
                        image_type = type
                        break
                else:
                    # skip invalid folders
                    continue

                if image_type != ImageTypeEnum.label:
                    tag_information = ExperimentReader._read_tag_information(
                        image_folder_path
                    )
                else:
                    tag_information = ExperimentReader._read_tag_information(
                        image_folder_path,
                        exclude_tags=exclude_tags,
                        break_on_unexpected_tag=False
                    )

                for tagged_image_folder_path, tags in tag_information:
                    image_tag = tags['tag'] if 'tag' in tags else None
                    if 'shrinkage' in tags:
                        shrinkage_level = int(
                            re.match(r'\d+', tags['shrinkage']).group(0)
                        )
                    else:
                        shrinkage_level = 1

                    for image_path in get_sub_paths(tagged_image_folder_path):
                        image_name = os.path.basename(
                            os.path.abspath(image_path)
                        )
                        match = re.search(
                            r"^\w*"
                            r"(?P<creation_date>\d{6})(?P<owner>\w{2})"
                            r"(?P<round_id>\d)(?P<plate_id>\w{2})_"
                            r"(?:\d+_)*"  # incl. images from SF with datetimes
                            r"(?P<well_name>[A-Z]\d{2})_"
                            r"T(?P<time_point>\d{4})F(?P<field_id>\d{3})"
                            r"L(?P<time_line_id>\d{2,3})A(?P<action_id>\d{2})"
                            r"Z(?P<z_stack>\d{2})C(?P<channel_id>\d{2})"
                            r"\.(tif|png)$",
                            image_name
                        )
                        if not match:
                            continue

                        # TODO: we miss the plate_id here (or the fixation_id?)
                        (
                            creation_date, owner, round_id, fixation_id,
                            well_name, time_point, field_id, time_line_id,
                            action_id, z_stack, channel_id, _
                        ) = match.groups()

                        # normalize well name
                        well_name = (
                            well_name[0:2].replace('0', '') + well_name[2:]
                        )

                        # TODO: we miss the channel_name here;
                        # requires meta information...
                        channel_id = int(channel_id)
                        if channel_id in channels:
                            channel = channels[channel_id]
                        else:
                            channel = Channel(
                                id=channel_id,
                                name="C%s" % str(channel_id).zfill(2)
                            )
                            channels[channel_id] = channel

                        if condition_df is not None:
                            treatment = \
                                condition_df.loc[well_name]['condition']
                            if treatment is None:
                                # TODO: print this warning only for the first
                                # image related to the mising well
                                print(
                                    (
                                        "WARNING: found missing treatment for "
                                        "well '%s' in experiment layout! "
                                        "No image information are read from "
                                        "this well."
                                    ) % (well_name,)
                                )
                                continue
                        else:
                            treatment = None

                        if well_name in wells:
                            well = wells[well_name]
                        else:
                            well = Well(
                                row_id=well_name[0].upper(),
                                column_id=int(well_name[1:3]),
                                treatment=treatment,
                                plate=plate
                            )
                            wells[well_name] = well

                        if image_type == ImageTypeEnum.tif:
                            mlf_image_idx = (
                                mlf_frame['file_name'] == image_name
                            )
                            if not any(mlf_image_idx):
                                raise Exception(
                                    (
                                        "Missing image '%s' in "
                                        "measurement data file!"
                                    ) % (image_name,)
                                )
                            mlf_image_row = mlf_frame[mlf_image_idx].iloc[0]
                            (ul_x, ul_y) = (
                                mlf_image_row['x_pixel'],
                                mlf_image_row['y_pixel']
                            )
                            (lr_x, lr_y) = (
                                ul_x + mlf_image_row['width'],
                                ul_y + mlf_image_row['height']
                             )
                        else:
                            bbox_microscope_space = \
                                well_bbox_df.loc[well_name]['bbox']
                            (ul_x, ul_y) = min(
                                bbox_microscope_space.exterior.coords
                            )
                            (lr_x, lr_y) = max(
                                bbox_microscope_space.exterior.coords
                            )

                        bbox_image_space = box(0, 0, lr_x-ul_x, lr_y-ul_y)
                        translation_matrix = np.array([
                            [1, 0, 0, ul_x/shrinkage_level],
                            [0, 1, 0, ul_y/shrinkage_level],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ], dtype=np.double)
                        scaling_matrix = np.array([
                            [shrinkage_level, 0, 0, 0],
                            [0, shrinkage_level, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ], dtype=np.double)
                        # @see: https://www.gaia-gis.it/fossil/libspatialite/
                        #           wiki?name=Affine+Transform
                        # note:
                        # the multiplication between two matrices is not a
                        # commutative operation: the relative order of operands
                        # is absolutely relevant.
                        image_matrix = ATMatrix(np.matmul(
                            scaling_matrix, translation_matrix
                        ))

                        image_channels = [ImageChannel(channel=channel)]
                        # TODO: extend code for RGBs (= multiple ImageChannels)
                        if image_type == ImageTypeEnum.rgb:
                            pass

                        image = Image(
                            name=image_name,
                            path=os.path.dirname(image_path),
                            type=image_type,
                            tag=image_tag,
                            image_matrix=image_matrix.as_binary(),
                            bbox=from_shape(bbox_image_space),
                            time_point=time_point,
                            field_id=field_id,
                            time_line_id=time_line_id,
                            z_stack=z_stack,
                            image_channels=image_channels,
                            action_id=action_id,
                            well=well
                        )

                stains = {
                    stain.name: stain for stain in Stain.query.all()
                }
                for well in wells.values():
                    treatment = well.treatment

                    if treatment is None:
                        continue

                    for row_idx, row in stain_df.loc[treatment].iterrows():
                        round_id, fixation_id, channel_id = row_idx

                        stain_name = row['stain_name']
                        if stain_name in stains:
                            stain = stains[stain_name]
                        else:
                            stain = Stain(
                                name=stain_name
                            )
                            stains[stain_name] = stain

                        # TODO: can we only apply stains to channels of
                        # TIF images?
                        for image_channel in (
                            ImageChannel.query
                            .join(Channel).filter(Channel.id == channel_id)
                            .join(Image)
                            .filter(Image.type == ImageTypeEnum.tif)
                            .join(Well).filter(Well.id == well.id)
                        ):
                            image_channel.stain = stain

    @staticmethod
    def _read_tag_information(
        folder_path, exclude_tags=[], break_on_unexpected_tag=True, tags={}
    ):
        sub_folder_paths = list(get_sub_paths(folder_path, dir_only=True))

        if len(sub_folder_paths) == 0:
            yield (folder_path, tags)

        for sub_folder_path in sub_folder_paths:
            sub_folder_name = os.path.basename(
                os.path.abspath(sub_folder_path)
            )
            if sub_folder_name in exclude_tags:
                continue

            tag = sub_folder_name
            tag_tuple = tag.split('_')
            if len(tag_tuple) != 2 or tag_tuple[0] == '':
                if break_on_unexpected_tag:
                    print(
                        (
                            "WARNING: found unexpected tag '%s' in %s! "
                            "Skip folder."
                        ) % (tag, os.path.abspath(sub_folder_path))
                    )
                    continue
                else:
                    # print(
                    #     "WARNING: found unexpected tag '%s' in %s."
                    #     % (tag, os.path.abspath(sub_folder_path))
                    # )
                    pass
            else:
                tags[tag_tuple[0]] = tag_tuple[1]
                # TODO: should we yield intermediate sub-folders as well?
                # yield (sub_folder_path, tags)

            yield from ExperimentReader._read_tag_information(
                sub_folder_path, exclude_tags, break_on_unexpected_tag, tags
            )

    @staticmethod
    def read_segmentation_information(experiment):
        session = Session()

        # Allow large images
        PILImage.MAX_IMAGE_PIXELS = np.inf

        for plate in (
            Plate.query
            .join(Experiment).filter(Experiment.id == experiment.id)
        ):
            for image in (
                Image.query.filter(Image.type == ImageTypeEnum.label)
                .join(Well)
                .join(Plate).filter(Plate.id == plate.id)
            ):
                # we assuem as a label image only has a single image channel
                image_channel = image.image_channels[0]

                if len(image_channel.image_channel_segmentations) > 0:
                    print(
                        (
                            "WARNING: found existing segmentations for "
                            "image '%s' [channel: %s]! Existing segmentations "
                            "are replaced."
                        ) % (image, image_channel)
                    )
                    for image_channel_segmentation in (
                        image_channel.image_channel_segmentations
                    ):
                        session.delete(image_channel_segmentation)

                # find related source images
                # hereby, the region_properties have to be extracted only once
                # for a label image as well as all related images
                related_image_path = image.path[
                    0:image.path.find(os.path.sep, len(plate.path)+1)
                ].replace("_SEG", "")

                # WARNING: does not consider any image tags; e.g. label image
                # may have tag1 while a 'related' MIP image have tag2
                if os.path.exists(related_image_path):
                    related_image_channels = (
                        ImageChannel.query
                        .join(Image).filter(
                            Image.path.like(related_image_path + '%') &
                            (Image.type != ImageTypeEnum.label)
                        )
                        .join(Well)
                        .join(Plate)
                        .join(Experiment)
                        .filter(Experiment.id == experiment.id)
                        .all()
                    )
                else:
                    related_image_channels = []

                pil_image = imread(os.path.join(image.path, image.name))
                region_properties = regionprops(pil_image)

                for property in region_properties:
                    ul_y, ul_x, lr_y, lr_x = property.bbox

                    label = property.label
                    pil_object_image = property.image.astype(int)

                    # edges of objects laying on an image border are not
                    # correctly detected by find_contours().
                    # hence, we pad the pil_object_image by 1 pixel.
                    padding = (1, 1)
                    pil_ext_object_image = pad(
                        pil_object_image,
                        padding, 'constant', constant_values=0
                    )

                    contour_ext = find_contours(pil_ext_object_image, 0)[0]
                    # find_contours returns an array of (y, x) coordinates
                    # while Shapely's Polygon expect (x, y) coordinates.
                    # hence, we have to swap the coordinates
                    contour_ext = contour_ext[:, [1, 0]]
                    # finally, we have to correct the contour coordinates which
                    # are shifted by the previous padding.
                    contour = contour_ext - padding[0]

                    translation_matrix = np.array([
                        [1, 0, 0, ul_x],
                        [0, 1, 0, ul_y],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ], dtype=np.double)
                    # @see: https://www.gaia-gis.it/fossil/libspatialite/
                    #           wiki?name=Affine+Transform
                    # note: The multiplication between two matrices is not a
                    # commutative operation: the relative order of operands is
                    # absolutely relevant.
                    segmentation_matrix = ATMatrix(translation_matrix)

                    segmentation = Segmentation(
                        contour=from_shape(Polygon(contour))
                    )
                    ImageChannelSegmentation(
                        label=label,
                        segmentation_matrix=segmentation_matrix.as_binary(),
                        image_channel=image_channel,
                        segmentation=segmentation
                    )

                    # note: we assume as the segmentation_matrix is the same
                    # for all related images!
                    for related_image_channel in related_image_channels:
                        ImageChannelSegmentation(
                            label=None,
                            segmentation_matrix=(
                                segmentation_matrix.as_binary()
                            ),
                            image_channel=related_image_channel,
                            segmentation=segmentation
                        )

    @staticmethod
    def read_segmentation_features(experiment):
        session = Session()

        local_image_channel_feature_types = {
            (
                feature_type.parent.name
                if feature_type.parent is not None else None,
                feature_type.name
            ): feature_type
            for feature_type in LocalImageChannelFeatureType.query.all()
        }
        shape_feature_types = {
            (
                feature_type.parent.name
                if feature_type.parent is not None else None,
                feature_type.name
            ): feature_type
            for feature_type in ShapeFeatureType.query.all()
        }

        for plate in (
            Plate.query.join(Experiment).filter(Experiment.id == experiment.id)
        ):
            for well in plate.wells:
                sfd_frame_cache = {}

                for image in well.images:
                    # WARNING: does not work for segmentations with tags,
                    # e.g.: ".../*_SEG/tag_1/features/features_*.csv"
                    # TODO: how to link TIF_MIP/tag_1/tag_2/ with
                    # .../TIF_MIP_SEG/tag_2/tag_1/features/features_*.csv
                    feature_file_path = os.path.abspath(
                        os.path.join(
                            "%s_SEG" % image.path[
                                0:(
                                    ("%s\\" % image.path)
                                    .find('\\', len(plate.path)+1)
                                )
                            ],
                            'features', "features_%s.csv" % (plate.bar_code,)
                        )
                    )
                    if not os.path.exists(feature_file_path):
                        continue

                    print(
                        "INFO: import features for plate '%s' and well '%s'."
                        % (plate.bar_code, well.name)
                    )

                    if feature_file_path in sfd_frame_cache:
                        sfd_frame = sfd_frame_cache[feature_file_path]
                    else:
                        sfd_frame = \
                            SegmentationFeatureReader.read(feature_file_path)
                        sfd_frame_cache[feature_file_path] = sfd_frame

                    if any(sfd_frame["well_name", "", ""].str.contains(
                        well.name, case=False, regex=False)
                    ):
                        sfd_frame = sfd_frame[
                            sfd_frame["well_name", "", ""].str.contains(
                                well.name, case=False, regex=False
                            )
                        ]
                    else:
                        print(
                            "WARNING: no features found for image '%s' in '%s'"
                            % (
                                image.path + os.path.sep + image.name,
                                feature_file_path
                            )
                        )
                        continue

                    # initialize image features & shape classes/types if they
                    # do not yet exist within the database
                    for feature_class in sfd_frame.feature_classes:
                        if feature_class == 'shape_descriptor':
                            if (
                                (None, feature_class) not in
                                shape_feature_types
                            ):
                                shape_feature_types[(None, feature_class)] = \
                                    ShapeFeatureType(name=feature_class)

                            for feature_type, channel_name in (
                                sfd_frame[feature_class].columns
                            ):
                                if (
                                    (feature_class, feature_type) not in
                                    shape_feature_types
                                ):
                                    shape_feature_types[
                                        (feature_class, feature_type)
                                    ] = (
                                        ShapeFeatureType(
                                            name=feature_type,
                                            parent=shape_feature_types[
                                                (None, feature_class)
                                            ]
                                        )
                                    )
                        else:
                            if (
                                (None, feature_class) not in
                                local_image_channel_feature_types
                            ):
                                local_image_channel_feature_types[
                                    (None, feature_class)
                                ] = \
                                    LocalImageChannelFeatureType(
                                        name=feature_class
                                    )

                            for feature_type, channel_name in (
                                sfd_frame[feature_class].columns
                            ):
                                if (
                                    (feature_class, feature_type) not in
                                    local_image_channel_feature_types
                                ):
                                    parent = (
                                        local_image_channel_feature_types[
                                            (None, feature_class)
                                        ]
                                    )
                                    local_image_channel_feature_types[
                                        (feature_class, feature_type)
                                    ] = (
                                        LocalImageChannelFeatureType(
                                            name=feature_type,
                                            parent=parent
                                        )
                                    )

                    if image.type == ImageTypeEnum.label:
                        for segmentation in (
                            Segmentation.query
                            .join(ImageChannelSegmentation)
                            .join(ImageChannel)
                            .join(Image).filter(Image.id == image.id)
                        ):
                            if len(segmentation.shape_features) > 0:
                                print(
                                    (
                                        "WARNING: found existing shape "
                                        "features for segmentation '%s'! "
                                        "Existing features are replaced."
                                    ) % segmentation
                                )
                                for shape_feature in (
                                    segmentation.shape_features
                                ):
                                    session.delete(shape_feature)

                            for feature_type, _ in (
                                sfd_frame['shape_descriptor']
                            ):
                                for row_idx, row in sfd_frame.iterrows():
                                    ShapeFeature(
                                        shape_feature_type=shape_feature_types[
                                            ('shape_descriptor', feature_type)
                                        ],
                                        value=row[
                                            'shape_descriptor',
                                            feature_type,
                                            ''
                                        ],
                                        segmentation=segmentation
                                    )
                    else:
                        for image_channel_segmentation in (
                            ImageChannelSegmentation.query
                            .join(ImageChannel)
                            .join(Image).filter(Image.id == image.id)
                        ):
                            # FIXME: we assume as each segmentation is
                            # associated with a single label image and a single
                            # label.
                            label_image_channel_segmentation = \
                                ImageChannelSegmentation.query.filter(
                                    ImageChannelSegmentation.label is not None
                                ).join(Segmentation).filter(
                                    Segmentation.id ==
                                    image_channel_segmentation.segmentation_id
                                ).first()
                            label = label_image_channel_segmentation.label

                            if len(
                                image_channel_segmentation
                                .local_image_channel_features
                            ) > 0:
                                print(
                                    (
                                        "WARNING: found existing local "
                                        "features for image channel "
                                        "segmentation '%s'! Existing features "
                                        "are replaced."
                                    ) % image_channel_segmentation
                                )
                                for local_image_channel_feature in (
                                    image_channel_segmentation
                                    .local_image_channel_features
                                ):
                                    session.delete(local_image_channel_feature)

                            row = sfd_frame[
                                (sfd_frame['object_number', '', ''] == label) &
                                (sfd_frame['well_name', '', ''] == well.name)
                            ]

                            for feature_class, feature_type, channel_name in (
                                row[sfd_frame.feature_classes]
                            ):
                                if len(row) == 0:
                                    print(
                                        "WARNING: No feature set found for "
                                        "label '%s'" % (label,)
                                    )
                                    continue
                                if len(row) > 1:
                                    raise RuntimeError(
                                        "Multiple feature sets found for "
                                        "label '%s'!" % (label,)
                                    )

                                if feature_class != 'shape_descriptor':
                                    LocalImageChannelFeature(
                                        local_image_channel_feature_type=(
                                            local_image_channel_feature_types[
                                                (feature_class, feature_type)
                                            ]
                                        ),
                                        value=(
                                            row.iloc[0][
                                                feature_class,
                                                feature_type,
                                                channel_name
                                            ]
                                        ),
                                        image_channel_segmentation=(
                                            image_channel_segmentation
                                        )
                                    )

    @staticmethod
    def read_segmentation_crops(experiment):
        """
        for image_segmentation in (
            ImageSegmentation.query
            .join(Image)
            .join(Well)
            .join(Plate)
            .join(Experiment)
            .filter(Experiment.id == experiment.id)
        ):
            image = image_segmentation.image
            roi_path = os.path.join(
                image.path, 'objs', "obj_%s" % image_segmentation.label
            )
        """
        pass
