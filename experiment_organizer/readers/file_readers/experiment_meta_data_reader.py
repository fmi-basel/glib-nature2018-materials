import os
import xlrd  # required by pandas for reading XLSX files
import pandas as pd


class ExperimentMetaDataReader():
    @staticmethod
    def read(meta_data_path):
        if meta_data_path is None or not os.path.exists(meta_data_path):
            raise RuntimeError(
                "The meta data file '%s' was not found!" % (meta_data_path,)
            )

        meta_data = {}
        meta_xlsx = pd.ExcelFile(meta_data_path)
        for sheet_name in meta_xlsx.sheet_names:
            if sheet_name.lower() == 'fixation':
                fixation_df = meta_xlsx.parse(sheet_name)
                fixation_df.set_index('id', inplace=True)
                meta_data['fixation'] = fixation_df
            else:
                pass

        return meta_data
