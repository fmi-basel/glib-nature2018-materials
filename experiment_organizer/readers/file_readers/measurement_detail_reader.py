import os
import pandas as pd
import xml.etree.cElementTree as ElementTree

from experiment_organizer.models import MeasurementDetailFrame
from experiment_organizer.utils import init_empty_dataframe


class MeasurementDetailReader():
    @staticmethod
    def read(measure_detail_path):
        if (
            measure_detail_path is None or
            not os.path.exists(measure_detail_path)
        ):
            raise RuntimeError(
                "The measurement detail file '%s' was not found!"
                % (measure_detail_path,)
            )

        if (
            measure_detail_path.endswith('xml') or
            measure_detail_path.endswith('mrf')
        ):
            mrf_frame = MeasurementDetailReader._read_xml(measure_detail_path)
        elif measure_detail_path.endswith('csv'):
            mrf_frame = MeasurementDetailReader._read_csv(measure_detail_path)
        else:
            raise RuntimeError(
                (
                    "Found unsupported file format '%s'!\n"
                    "Expected formats: XML, MRF, CSV."
                ) % (measure_detail_path,)
            )

        return mrf_frame

    @staticmethod
    def _read_csv(measure_detail_path):
        csv_frame = pd.read_csv(measure_detail_path)
        no_records = len(csv_frame)

        mrf_frame = MeasurementDetailFrame(
            index=range(0, no_records), file_path=measure_detail_path
        )

        for column in csv_frame.columns:
            mrf_frame[column] = csv_frame[column]

        # debug: print empty columns
        # print(
        #     "DEBUG: empty columns in mrf_frame\n%s"
        #     % mrf_frame.columns[mrf_frame.isna().all()].tolist()
        # )

        return mrf_frame

    @staticmethod
    def _read_xml(measure_detail_path):
        mrf_xml = ElementTree.parse(measure_detail_path).getroot()
        ns = {"bts": "http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"}
        no_records = len(
            mrf_xml.findall("bts:MeasurementChannel", namespaces=ns)
        )

        mrf_frame = MeasurementDetailFrame(
            index=range(0, no_records), file_path=measure_detail_path
        )

        for idx, record in enumerate(
            mrf_xml.findall("bts:MeasurementChannel", namespaces=ns)
        ):
            mrf_frame.iloc[idx] = (
                int(record.get("{%s}Ch" % ns["bts"])),
                float(record.get("{%s}HorizontalPixelDimension" % ns["bts"])),
                float(record.get("{%s}VerticalPixelDimension" % ns["bts"])),
                int(record.get("{%s}CameraNumber" % ns["bts"])),
                int(record.get("{%s}InputBitDepth" % ns["bts"])),
                int(record.get("{%s}InputLevel" % ns["bts"])),
                int(record.get("{%s}HorizontalPixels" % ns["bts"])),
                int(record.get("{%s}VerticalPixels" % ns["bts"])),
                int(record.get("{%s}FilterWheelPosition" % ns["bts"])),
                int(record.get("{%s}FilterPosition" % ns["bts"])),
                record.get("{%s}ShadingCorrectionSource" % ns["bts"])
            )

        return mrf_frame
