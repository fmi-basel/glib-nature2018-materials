import os
import pandas as pd
import re

from experiment_organizer.models import SegmentationFeatureFrame


class SegmentationFeatureReader():
    @staticmethod
    def read(segmentation_feature_path):
        if (
            segmentation_feature_path is None or
            not os.path.exists(segmentation_feature_path)
        ):
            raise RuntimeError(
                "The segmentation feature file '%s' was not found!"
                % (segmentation_feature_path,)
            )

        if segmentation_feature_path.endswith('csv'):
            sfd_frame = SegmentationFeatureReader._read_csv(
                segmentation_feature_path
            )
        else:
            raise RuntimeError(
                "Found unsupported file format '%s'!\nExpected formats: CSV."
                % (segmentation_feature_path,)
            )

        return sfd_frame

    @staticmethod
    def _read_csv(segmentation_feature_path):
        # TODO: standardize separator? (e.g. ',', '\t')
        csv_frame = pd.read_csv(
            segmentation_feature_path, sep=',', encoding='utf-8'
        )
        csv_frame.rename(
            index=str,
            columns={"well": "well_name", "label": "object_number"},
            inplace=True
        )
        no_records = len(csv_frame)

        # normalize well_names
        csv_frame['well_name'] = csv_frame['well_name'].apply(
            lambda well_name: well_name[0:2].replace('0', '') + well_name[2:]
        )

        sfd_frame = SegmentationFeatureFrame(
            index=range(0, no_records),
            file_path=segmentation_feature_path
        )

        for column in csv_frame.columns:
            sfd_frame[column] = list(csv_frame[column])

        # create multi-index over feature classes
        index_tuples = []
        feature_classes = [
            'shape_descriptor', 'zernike_feature',
            'texture_haralick', 'texture_local_bin_pattern',
            'intensity'
        ]
        invalid_columns = []
        for column in sfd_frame.columns:
            for feature_class in feature_classes:
                if column.startswith(feature_class):
                    if feature_class == 'shape_descriptor':
                        match = re.search(r".*_([c|C]\d{1,2})$", column)
                        if match:
                            print(
                                (
                                    "WARNING: shape descriptor '%s' contains "
                                    "channel information which are ignored!"
                                ) % (column,)
                            )
                            channel = match[1]
                            feature_type = \
                                column[len(feature_class)+1:].replace(
                                    '_' + channel, ''
                                )
                        else:
                            feature_type = column[len(feature_class)+1:]
                        index_tuples.append((feature_class, feature_type, ''))
                    else:
                        match = re.search(r".*_([c|C]\d{1,2})$", column)
                        if not match:
                            print(
                                (
                                    "WARNING: ignore feature '%s' due missing "
                                    "channel information!"
                                ) % (column,)
                            )
                            invalid_columns.append(column)
                            break
                        channel = match[1]
                        feature_type = column[len(feature_class)+1:].replace(
                            '_' + channel, ''
                        )
                        index_tuples.append(
                            (feature_class, feature_type, channel)
                        )
                    break
            else:
                index_tuples.append((column, '', ''))

        sfd_frame.drop(invalid_columns, axis=1, inplace=True)

        feature_idx = pd.MultiIndex.from_tuples(
            index_tuples,
            names=['feature_class', 'feature_type', 'channel_name']
        )
        sfd_frame.columns = feature_idx

        for feature_class in feature_classes:
            if feature_class in sfd_frame:
                sfd_frame.feature_classes.append(feature_class)
            else:
                print(
                    "WARNING: missing feature class '%s' in '%s'!"
                    % (feature_class, segmentation_feature_path)
                )

        # debug: print empty columns
        # print(
        #     "DEBUG: empty columns in sfd_frame\n%s"
        #     % sfd_frame.columns[sfd_frame.isna().all()].tolist()
        # )

        return sfd_frame
