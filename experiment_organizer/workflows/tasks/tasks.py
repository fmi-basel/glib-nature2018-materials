import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy.sql import select, cast, join, func, literal
from sqlalchemy.orm import aliased

from experiment_organizer.models import *
from experiment_organizer.views import *

from experiment_organizer.workflows.tasks.base import Task


class NOPTask(Task):
    def run(self, data):
        return data


class FindOverlappingRoundSegmentationsTask(Task):
    def run(self, experiment):
        session = Session()

        print(
            "INFO: find overlapping round segmentations for experiment %s."
            % experiment
        )

        rows = session.query(
            select([WellView]).where(WellView.experiment_id == experiment.id)
        ).all()
        well_df = pd.DataFrame(rows)

        if len(well_df) == 0:
            print(
                "WARNING: No data to process for experiment %s!" % experiment
            )
            return []

        for (fixation_name, fixation_time_point, well_name), well_group in (
            well_df.groupby(
                ['fixation_name', 'fixation_time_point', 'well_name']
            )
        ):
            # estimate shift using point set registration
            # implementation: Iterative Closest Point (ICP)
            # @see https://en.wikipedia.org/wiki/Point_set_registration
            well_rounds = np.sort(
                well_group['plate_round_id'].unique()
            )

            # TODO: add tag as filter criterion
            filtered_image_channel_segmentation_view = select(
                [ImageChannelSegmentationView]
            ).where(
                (ImageChannelSegmentationView.experiment_id == experiment.id) &
                (ImageChannelSegmentationView.fixation_name == fixation_name) &
                (ImageChannelSegmentationView.well_name == well_name) &
                (
                    ImageChannelSegmentationView.image_type ==
                    ImageTypeEnum.label
                )
                # & (ImageChannelSegmentationView.image_tag == '1')
            )
            source = aliased(filtered_image_channel_segmentation_view)
            target = aliased(filtered_image_channel_segmentation_view)
            source_segmentation_contour_ms = (
                source.c.image_channel_segmentation_contour_microscope_space
            )
            target_segmentation_contour_ms = (
                target.c.image_channel_segmentation_contour_microscope_space
            )
            source_segmentation_contour_cs = (
                source.c.image_channel_segmentation_contour_segmentation_space
            )
            target_segmentation_contour_cs = (
                target.c.image_channel_segmentation_contour_segmentation_space
            )

            # to improve performance, we use the Manhattan distance
            manhattan_dist_segmentation_space = (
                func.abs(
                    func.X(func.Centroid(source_segmentation_contour_ms)) -
                    func.X(func.Centroid(target_segmentation_contour_ms))
                )
            ) + (
                func.abs(
                    func.Y(func.Centroid(source_segmentation_contour_ms)) -
                    func.Y(func.Centroid(target_segmentation_contour_ms))
                )
            )

            ref_round_id = well_rounds[0]
            round_shifts = [
                # the reference round has a shift of (0,0) in respect to itself
                (ref_round_id, 0, 0)
            ]
            samples = 100
            for target_round_id in well_rounds[1:]:
                select_stmt = select([
                    source.c.image_channel_segmentation_label.label(
                        'source_segmentation_label'
                    ),
                    source.c.plate_round_id.label('source_round_id'),
                    target.c.image_channel_segmentation_label.label(
                        'target_segmentation_label'
                    ),
                    target.c.plate_round_id.label('target_round_id'),
                    (
                        func.X(func.Centroid(source_segmentation_contour_ms)) -
                        func.X(func.Centroid(target_segmentation_contour_ms))
                    ).label('x_shift'),
                    (
                        func.Y(func.Centroid(source_segmentation_contour_ms)) -
                        func.Y(func.Centroid(target_segmentation_contour_ms))
                    ).label('y_shift'),
                    func.min(manhattan_dist_segmentation_space).label(
                        'min_manhattan_dist'
                    )
                ]).select_from(
                    join(
                        source, target,
                        # ST_Expand already returns the extended bounding-box
                        # of the contour, hence, no manual calculateion of the
                        # bbox is needed
                        (
                            func.ST_Intersects(
                                func.ST_Expand(
                                    source_segmentation_contour_ms, 100
                                ),
                                func.ST_Expand(
                                    target_segmentation_contour_ms, 100
                                )
                            ) == 1
                        ) &
                        (
                            (source.c.plate_round_id == int(ref_round_id)) &
                            (target.c.plate_round_id == int(target_round_id))
                        )
                    )
                ).group_by(
                    source.c.image_channel_segmentation_id
                ).limit(
                    samples
                )

                round_shift_frame = pd.DataFrame(
                    session.query(select_stmt).all()
                )

                if len(round_shift_frame) == 0:
                    continue
                elif len(round_shift_frame) == 1:
                    round_shifts.append(
                        (
                            target_round_id,
                            round_shift_frame['x_shift'].mean(),
                            round_shift_frame['y_shift'].mean()
                        )
                    )
                else:
                    round_shifts.append(
                        (
                            target_round_id,
                            round_shift_frame['x_shift'].sort_values()[
                                int(len(round_shift_frame)*1/5):
                                int(len(round_shift_frame)*4/5)
                            ].mean(),
                            round_shift_frame['y_shift'].sort_values()[
                                int(len(round_shift_frame)*1/5):
                                int(len(round_shift_frame)*4/5)
                            ].mean()
                        )
                    )

            # initialize temporary table
            # @see https://stackoverflow.com/questions/44140632/
            #          use-temp-table-with-sqlalchemy
            temp_round_shifts_table = [
                select([
                    cast(literal(int(round_id)), sqlalchemy.Integer).label(
                        "round_id"
                    ),
                    cast(literal(x_shift), sqlalchemy.Float).label("shift_x"),
                    cast(literal(y_shift), sqlalchemy.Float).label("shift_y"),
                ])
                for round_id, x_shift, y_shift in round_shifts
            ]
            temp_round_shifts_table = sqlalchemy.union_all(
                *temp_round_shifts_table
            )

            manhattan_dist_microscope_space = (
                func.abs(
                    (
                        func.X(func.Centroid(source_segmentation_contour_ms)) +
                        select([temp_round_shifts_table.c.shift_x]).where(
                            temp_round_shifts_table.c.round_id ==
                            source.c.plate_round_id
                        )
                    ) -
                    (
                        func.X(func.Centroid(target_segmentation_contour_ms)) +
                        select([temp_round_shifts_table.c.shift_x]).where(
                            temp_round_shifts_table.c.round_id ==
                            target.c.plate_round_id
                        )
                    )
                )
            ) + (
                func.abs(
                    (
                        func.Y(func.Centroid(source_segmentation_contour_ms)) +
                        select([temp_round_shifts_table.c.shift_y]).where(
                            temp_round_shifts_table.c.round_id ==
                            source.c.plate_round_id
                        )
                    ) -
                    (
                        func.Y(func.Centroid(target_segmentation_contour_ms)) +
                        select([temp_round_shifts_table.c.shift_y]).where(
                            temp_round_shifts_table.c.round_id ==
                            target.c.plate_round_id
                        )
                    )
                )
            )

            # TODO: Extract the manhattan_dist_microscope_space which is
            # currently computed twice during the select query.
            select_stmt = select([
                source.c.image_channel_segmentation_id.label(
                    'source_segmentation_id'
                ),
                source.c.plate_round_id.label('source_round_id'),
                source.c.image_channel_segmentation_label.label(
                    'source_segmentation_label'
                ),
                target.c.image_channel_segmentation_id.label(
                    'target_segmentation_id'
                ),
                target.c.plate_round_id.label('target_round_id'),
                target.c.image_channel_segmentation_label.label(
                    'target_segmentation_label'
                ),
                (
                    func.Area(
                        func.Intersection(
                            # Using a zero-width buffer (ST_Buffer) cleans up
                            # many topology problems such as self-intersections
                            func.ST_Buffer(
                                source_segmentation_contour_cs, 0.0
                            ),
                            func.ST_Buffer(target_segmentation_contour_cs, 0.0)
                        )
                    ) / func.Area(
                        func.GUnion(
                            func.ST_Buffer(
                                source_segmentation_contour_cs, 0.0
                            ),
                            func.ST_Buffer(target_segmentation_contour_cs, 0.0)
                        )
                    ) * func.Exp(-0.001 * manhattan_dist_microscope_space)
                ).label('similarity_score')
            ]).select_from(
                join(
                    source, target,
                    (manhattan_dist_microscope_space < 200) &
                    (
                        source.c.plate_round_id < target.c.plate_round_id
                        # implication:
                        # source != target
                        # source.plate_round_id != target.plate_round_id
                        # no duplicate entries for (target.id=1, source.id=2)
                        #     and (target.id=2, source.id=1)
                    ),
                    # do a left-outer-join
                    # (hereby, we fix a source and add a target if an
                    # intersection was found, None otherwise)
                    isouter=True
                )
            )

            # returns a list where each image segmentation is enlistet at least
            # once as a source_segmentation_id.
            # source segmentations without any overlapping target are enlisted
            # in a single row with their target_segmentation_id set to "None"
            rows = session.query(select_stmt).all()
            data_frame = pd.DataFrame(rows)
            data_frame.fillna(value=np.nan, inplace=True)

            yield {
                'data_partition': {
                    'fixation_name': fixation_name,
                    'well_name': well_name
                },
                'data_frame': data_frame
            }


class FindBestOverlappingSegmentationLinksTask(Task):
    def __init__(self, location_field):
        self.location_field = location_field

    def run(self, input):
        data_partition = input['data_partition']
        data_frame = input['data_frame']

        print(
            "INFO: find best overlapping segmentation links for partition %s."
            % data_partition
        )

        if len(data_frame) == 0:
            print(
                "WARNING: No data to process for partition %s!"
                % data_partition
            )
            return []

        data_frame['distance'] = data_frame.apply(
            lambda row: (
                row['target_%s' % self.location_field] -
                row['source_%s' % self.location_field]
            ),
            axis=1
        )

        # we do not need to filter any rows here - e.g. by the similarity_score
        # as this happens automatically while searching for the best match
        source_groups = data_frame.groupby('source_segmentation_id')
        for source_segmentation_id, target_group in source_groups:
            target_group = target_group.sort_values(
                ['distance', 'similarity_score'], ascending=[True, False]
            )
            # remove all but the closest target (in respect to the location)
            # if multiple targets within the same distance are found
            # the highest similarity_score wins.
            data_frame.drop(target_group.index[1:], inplace=True)

        target_groups = data_frame.groupby('target_segmentation_id')
        for target_segmentation_id, source_group in target_groups:
            source_group = source_group.sort_values(
                ['distance', 'similarity_score'], ascending=[True, False]
            )
            # remove all but the closest source (in respect to the location)
            # if multiple sources within the same distance are found
            # the highest similarity_score wins.
            for row_idx, row in source_group[1:].iterrows():
                data_frame.at[row_idx, 'target_segmentation_id'] = np.nan
                data_frame.at[row_idx, 'target_round_id'] = np.nan
                data_frame.at[row_idx, 'target_segmentation_label'] = np.nan
                data_frame.at[row_idx, 'similarity_score'] = np.nan
                data_frame.at[row_idx, 'distance'] = np.nan

        return [{
            'data_partition': data_partition,
            'data_frame': data_frame
        }]


class LinkOverlappingSegmentationsTask(Task):
    def __init__(self, location_field, object_type, min_similariry_score=0):
        self.location_field = location_field

        if object_type == ObjectTypeEnum.organoid:
            self.object_type = ObjectTypeEnum.organoid
            self.object_cls = Organoid
        elif object_type == ObjectTypeEnum.cell:
            self.object_type = ObjectTypeEnum.cell
            self.object_cls = Cell
        else:
            raise RuntimeError("Object type '%s' is unknown!" % (object_type,))

        self.min_similariry_score = min_similariry_score

    def run(self, input):
        session = Session()

        data_partition = input['data_partition']
        data_frame = input['data_frame']

        print(
            "INFO: link overlapping segmentations for partition %s."
            % data_partition
        )

        if len(data_frame) == 0:
            print(
                "WARNING: No data to process for partition %s!"
                % data_partition
            )
            return []

        data_frame['index'] = data_frame['source_segmentation_id']
        data_frame.set_index('index', inplace=True)
        data_frame['object_id'] = np.nan
        data_frame = data_frame.sort_values(
            'source_%s' % self.location_field, ascending=True
        )

        for row_idx in data_frame.index:
            row = data_frame.loc[row_idx]
            # we have to convert the datatypes as loc[] involves an
            # (unexpected) internal cast ...
            source_segmentation_id = int(row['source_segmentation_id'])
            target_segmentation_id = (
                np.nan if np.isnan(row['target_segmentation_id'])
                else int(row['target_segmentation_id'])
            )
            object_id = (
                np.nan if np.isnan(row['object_id'])
                else int(row['object_id'])
            )

            # found first image channel segmentation of linking chain
            if np.isnan(object_id):
                object = Object(type=self.object_type)
                object.add()
                concrete_object = self.object_cls(object=object)
                concrete_object.add()
                object_id = object.id
                for segmentation in (
                    session.query(Segmentation)
                    .join(ImageChannelSegmentation)
                    .filter(
                        ImageChannelSegmentation.id == source_segmentation_id
                    )
                ):
                    segmentation.object = Object.query.filter(
                        Object.id == object_id
                    ).one()
                data_frame.at[source_segmentation_id, 'object_id'] = object_id

                if np.isnan(target_segmentation_id):
                    print(
                        (
                            "WARNING: No linking found for image segmentation "
                            "'%s'!"
                        ) % (source_segmentation_id,)
                    )

            # add link to the next image channel segmentation
            if not np.isnan(target_segmentation_id):
                if not np.isnan(
                    data_frame.at[target_segmentation_id, 'object_id']
                ):
                    print(
                        (
                            "WARNING: Multiple links found for image "
                            "segmentation '%s'! Skip additional assignments."
                        ) % (target_segmentation_id,)
                    )
                else:
                    if row['similarity_score'] < self.min_similariry_score:
                        print(
                            (
                                "WARNING: Skip linking of segmentations '%s' "
                                "and '%s' because the similarity score (%s) "
                                "is lower than the threshold (%s)."
                            ) % (
                                source_segmentation_id,
                                target_segmentation_id,
                                row['similarity_score'],
                                self.min_similariry_score
                            )
                        )
                    else:
                        for segmentation in (
                            session.query(Segmentation)
                            .join(ImageChannelSegmentation)
                            .filter(
                                ImageChannelSegmentation.id ==
                                target_segmentation_id
                            )
                        ):
                            segmentation.object = Object.query.filter(
                                Object.id == object_id
                            ).one()

                        data_frame.at[target_segmentation_id, 'object_id'] = \
                            data_frame.at[source_segmentation_id, 'object_id']

        return [{
            'data_partition': data_partition,
            'data_frame': data_frame
        }]


class FindOverlappingStackSegmentationsTask(Task):
    def run(self, experiment):
        session = Session()

        print(
            "INFO: find overlapping stack segmentations for experiment %s."
            % experiment
        )

        rows = session.query(
            select([WellView]).where(WellView.experiment_id == experiment.id)
        ).all()
        well_df = pd.DataFrame(rows)

        if len(well_df) == 0:
            print(
                "WARNING: No data to process for experiment %s!" % experiment
            )
            return []

        for plate_round_id, fixation_name, fixation_time_point, well_name in (
            well_df.groupby(
                [
                    'plate_round_id', 'fixation_name', 'fixation_time_point',
                    'well_name'
                ]
            ).groups
        ):
            # TODO: add tag as filter criterion
            filtered_image_channel_segmentation_view = select(
                [ImageChannelSegmentationView]
            ).where(
                (ImageChannelSegmentationView.experiment_id == experiment.id) &
                (
                    ImageChannelSegmentationView.plate_round_id ==
                    plate_round_id
                ) &
                (ImageChannelSegmentationView.fixation_name == fixation_name) &
                (ImageChannelSegmentationView.well_name == well_name) &
                (
                    ImageChannelSegmentationView.image_type ==
                    ImageTypeEnum.label
                )
                # & (ImageChannelSegmentationView.image_tag == '1')
            )
            source = aliased(filtered_image_channel_segmentation_view)
            target = aliased(filtered_image_channel_segmentation_view)
            source_segmentation_contour_ms = (
                source.c.image_channel_segmentation_contour_microscope_space
            )
            target_segmentation_contour_ms = (
                target.c.image_channel_segmentation_contour_microscope_space
            )
            source_segmentation_contour_cs = (
                source.c.image_channel_segmentation_contour_segmentation_space
            )
            target_segmentation_contour_cs = (
                target.c.image_channel_segmentation_contour_segmentation_space
            )

            select_stmt = select([
                source.c.image_channel_segmentation_id.label(
                    'source_segmentation_id'
                ),
                source.c.image_z_stack.label('source_image_z_stack'),
                source.c.image_channel_segmentation_label.label(
                    'source_segmentation_label'
                ),
                target.c.image_channel_segmentation_id.label(
                    'target_segmentation_id'
                ),
                target.c.image_z_stack.label('target_image_z_stack'),
                target.c.image_channel_segmentation_label.label(
                    'target_segmentation_label'
                ),
                (
                    func.Area(
                        func.Intersection(
                            # Using a zero-width buffer (ST_Buffer) cleans up
                            # many topology problems such as self-intersections
                            func.ST_Buffer(
                                source_segmentation_contour_cs, 0.0
                            ),
                            func.ST_Buffer(target_segmentation_contour_cs, 0.0)
                        )
                    ) / func.Area(
                        func.GUnion(
                            func.ST_Buffer(
                                source_segmentation_contour_cs, 0.0
                            ),
                            func.ST_Buffer(target_segmentation_contour_cs, 0.0)
                        )
                    ) * func.Exp(
                        -0.05 * func.ST_Distance(
                            source_segmentation_contour_ms,
                            target_segmentation_contour_ms
                        )
                    )
                ).label('similarity_score')
            ]).select_from(
                join(
                    source, target,
                    # ST_Expand already returns the extended bounding-box of
                    # the contour, hence, no manual calculateion of the bbox
                    # is needed
                    (
                        func.ST_Intersects(
                            func.ST_Expand(
                                source_segmentation_contour_ms, 100
                            ),
                            func.ST_Expand(target_segmentation_contour_ms, 100)
                        ) == 1
                    ) &
                    (target.c.image_z_stack - source.c.image_z_stack <= 3) &
                    (
                        source.c.image_z_stack < target.c.image_z_stack
                        # implication:
                        # source != target
                        # source.image_z_stack != target.image_z_stack
                        # no duplicate entries for (target.id=1, source.id=2)
                        #     and (target.id=2, source.id=1)
                    ),
                    # do a left-outer-join
                    # (hereby, we fix a source and add a target if an
                    # intersection was found, None otherwise)
                    isouter=True
                )
            )

            # returns a list where each image segmentation is enlistet at least
            # once as a source_segmentation_id.
            # source segmentations without any overlapping target are enlisted
            # in a single row with their target_segmentation_id set to "None"
            rows = session.query(select_stmt).all()
            data_frame = pd.DataFrame(rows)
            data_frame.fillna(value=np.nan, inplace=True)

            yield {
                'data_partition': {
                    'plate_round_id': plate_round_id,
                    'fixation_name': fixation_name,
                    'well_name': well_name
                },
                'data_frame': data_frame
            }
