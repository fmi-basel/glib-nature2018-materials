import os
import numpy as np
import pandas as pd
import xml.etree.cElementTree as ElementTree

from experiment_organizer.models import MeasurementDataFrame
from experiment_organizer.readers.file_readers import MeasurementDetailReader
from experiment_organizer.utils import init_empty_dataframe


class MeasurementDataReader():
    @staticmethod
    def read(
        measurement_data_path, measurement_detail_path=None, load_details=True
    ):
        if (
            measurement_data_path is None or
            not os.path.exists(measurement_data_path)
        ):
            raise RuntimeError(
                "The measurement data file '%s' was not found!"
                % (measurement_data_path,)
            )

        if (
            measurement_data_path.endswith('xml') or
            measurement_data_path.endswith('mlf')
        ):
            mlf_frame = MeasurementDataReader._read_xml(
                measurement_data_path,
                {
                    'file_path': os.path.abspath(
                        os.path.join(measurement_data_path, '..', 'TIF')
                    )
                }
            )
        elif measurement_data_path.endswith('csv'):
            mlf_frame = MeasurementDataReader._read_csv(
                measurement_data_path,
                {
                    'file_path': os.path.abspath(
                        os.path.join(measurement_data_path, '..', 'TIF')
                    )
                }
            )
        else:
            raise RuntimeError(
                (
                    "Found unsupported file format '%s'!\n"
                    "Expected formats: XML, MLF, CSV."
                ) % (measurement_data_path,)
            )

        if load_details:
            if (
                measurement_detail_path is None and
                os.path.exists(
                    os.path.join(
                        os.path.dirname(measurement_data_path),
                        'MeasurementDetail.mrf'
                    )
                )
            ):
                measurement_detail_path = os.path.join(
                    os.path.dirname(measurement_data_path),
                    'MeasurementDetail.mrf'
                )

            mrf_frame = MeasurementDetailReader.read(measurement_detail_path)
            MeasurementDataReader.read_detail_information(mlf_frame, mrf_frame)

        return mlf_frame

    @staticmethod
    def _read_csv(measurement_data_path, defaults={}):
        csv_frame = pd.read_csv(
            measurement_data_path, sep='\t', encoding='utf-8'
        )
        no_records = len(csv_frame)

        mlf_frame = MeasurementDataFrame(
            index=range(0, no_records), file_path=measurement_data_path
        )
        for column in defaults:
            mlf_frame[column] = defaults[column]

        for column in csv_frame.columns:
            mlf_frame[column] = csv_frame[column]

        # debug: print empty columns
        # print(
        #     "DEBUG: empty columns in mlf_frame\n%s"
        #     % mlf_frame.columns[mlf_frame.isna().all()].tolist()
        # )

        return mlf_frame

    @staticmethod
    def _read_xml(measurement_data_path, defaults={}):
        mlf_xml = ElementTree.parse(measurement_data_path).getroot()
        ns = {"bts": "http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"}
        no_records = len(
            mlf_xml.findall("bts:MeasurementRecord", namespaces=ns)
        )
        first_element = mlf_xml.find("bts:MeasurementRecord", namespaces=ns)

        mlf_frame = MeasurementDataFrame(
            index=range(0, no_records), file_path=measurement_data_path
        )
        for column in defaults:
            mlf_frame[column] = defaults[column]

        for idx, record in enumerate(
            mlf_xml.findall("bts:MeasurementRecord", namespaces=ns)
        ):
            rec_type = record.get("{%s}Type" % ns["bts"])
            if rec_type == "ERR":
                mlf_frame.iat[idx, 0] = rec_type
                print(
                    (
                        "WARNING: "
                        "Found an 'ERR' entry at line '%s' in '%s'!\n"
                        "The entry is skipped."
                    ) % (measurement_data_path, idx)
                )
                continue

            partial_tile_id = record.get("{%s}PartialTileIndex" % ns["bts"])
            partial_tile_id = (
                int(partial_tile_id) if partial_tile_id is not None else None
            )
            time = pd.to_datetime(record.get("{%s}Time" % ns["bts"]))
            time = pd.to_datetime(time) if time is not None else None

            well_row_id = record.get("{%s}Row" % ns["bts"])
            well_col_id = record.get("{%s}Column" % ns["bts"])
            well_name = chr(64+int(well_row_id)) + well_col_id
            time_point = int(record.get("{%s}TimePoint" % ns["bts"]))
            field_id = int(record.get("{%s}FieldIndex" % ns["bts"]))

            tile_x_id = record.get("{%s}TileXIndex" % ns["bts"])
            tile_x_id = int(tile_x_id) if tile_x_id is not None else None
            tile_y_id = record.get("{%s}TileYIndex" % ns["bts"])
            tile_y_id = int(tile_y_id) if tile_y_id is not None else None
            z_index = int(record.get("{%s}ZIndex" % ns["bts"]))

            timeline_id = int(record.get("{%s}TimelineIndex" % ns["bts"]))
            action_id = int(record.get("{%s}ActionIndex" % ns["bts"]))
            action = record.get("{%s}Action" % ns["bts"])

            # we mirror the y coordinate to fit with the field layout
            x_micrometer = float(record.get("{%s}X" % ns["bts"]))
            y_micrometer = -float(record.get("{%s}Y" % ns["bts"]))
            z_micrometer = float(record.get("{%s}Z" % ns["bts"]))

            channel_id = int(record.get("{%s}Ch" % ns["bts"]))

            # we use iat[] here for (significant) performance reasons
            mlf_frame.iat[idx, 0] = rec_type
            mlf_frame.iat[idx, 1] = time
            mlf_frame.iat[idx, 2] = well_name
            mlf_frame.iat[idx, 3] = int(well_col_id)
            mlf_frame.iat[idx, 4] = int(well_row_id)
            mlf_frame.iat[idx, 5] = time_point
            mlf_frame.iat[idx, 6] = field_id
            mlf_frame.iat[idx, 7] = partial_tile_id
            mlf_frame.iat[idx, 8] = tile_x_id
            mlf_frame.iat[idx, 9] = tile_y_id
            mlf_frame.iat[idx, 10] = z_index
            mlf_frame.iat[idx, 11] = timeline_id
            mlf_frame.iat[idx, 12] = action_id
            mlf_frame.iat[idx, 13] = action
            mlf_frame.iat[idx, 14] = x_micrometer
            mlf_frame.iat[idx, 15] = y_micrometer
            mlf_frame.iat[idx, 16] = z_micrometer
            # mlf_frame.iat[idx, 17] = np.nan  # x_pixel
            # mlf_frame.iat[idx, 18] = np.nan  # y_pixel
            # mlf_frame.iat[idx, 19] = np.nan  # bit_depth
            # mlf_frame.iat[idx, 20] = np.nan  # width
            # mlf_frame.iat[idx, 21] = np.nan  # height
            mlf_frame.iat[idx, 22] = channel_id
            # mlf_frame.iat[idx, 23] = np.nan  # camera_no
            # mlf_frame.iat[idx, 24] = np.nan  # file_path
            mlf_frame.iat[idx, 25] = record.text  # file_name

        return mlf_frame

    @staticmethod
    def read_detail_information(mlf_frame, mrf_frame):
        for mlf_idx in range(len(mlf_frame)):

            if mlf_frame.iat[mlf_idx, 0] == 'ERR':
                continue

            mrf_idx = mrf_frame[
                mrf_frame['channel_id'] == mlf_frame.iat[mlf_idx, 22]
            ].index[0]

            x_pixel = (
                # mlf_row.x_micrometer / mrf_row.horizontal_pixels
                np.ceil(mlf_frame.iat[mlf_idx, 14] / mrf_frame.iat[mrf_idx, 1])
            )
            y_pixel = (
                # mlf_row.y_micrometer / mrf_row.vertical_pixels
                np.ceil(mlf_frame.iat[mlf_idx, 15] / mrf_frame.iat[mrf_idx, 2])
            )
            bit_depth = (
                # mrf_row.input_bit_depth
                int(mrf_frame.iat[mrf_idx, 4])
            )
            width = (
                # mrf_row.horizontal_pixels
                int(mrf_frame.iat[mrf_idx, 6])
            )
            height = (
                # mrf_row.vertical_pixels
                int(mrf_frame.iat[mrf_idx, 7])
            )
            camera_no = (
                # mrf_row.camera_no
                int(mrf_frame.iat[mrf_idx, 3])
            )

            mlf_frame.iat[mlf_idx, 17] = x_pixel
            mlf_frame.iat[mlf_idx, 18] = y_pixel
            mlf_frame.iat[mlf_idx, 19] = bit_depth
            mlf_frame.iat[mlf_idx, 20] = width
            mlf_frame.iat[mlf_idx, 21] = height
            mlf_frame.iat[mlf_idx, 23] = camera_no
