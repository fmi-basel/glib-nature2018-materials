import datetime
import numpy as np
import os
import xlrd  # required by pandas for reading XLSX files
import pandas as pd

from experiment_organizer.models import ExperimentLayout


class ExperimentLayoutReader():
    @staticmethod
    def read(experiment_layout_path):
        experiment_layout = ExperimentLayout()

        if (
            experiment_layout_path is None or
            not os.path.exists(experiment_layout_path)
        ):
            raise RuntimeError(
                "The experiment layout file '%s' was not found!"
                % (experiment_layout_path,)
            )

        layout_xlsx = pd.ExcelFile(experiment_layout_path)
        for sheet_name in layout_xlsx.sheet_names:
            sheet = layout_xlsx.parse(sheet_name, header=None, index_col=None)
            condition_df, stain_df = \
                ExperimentLayoutReader._read_plate_sheet(sheet)

            if sheet_name.lower() == 'default':
                sheet_name = 'default'  # force lower-case

            experiment_layout.plate_condition_map[sheet_name] = condition_df
            experiment_layout.plate_stain_map[sheet_name] = stain_df

        return experiment_layout

    @staticmethod
    def _read_plate_sheet(sheet):
        # TODO: do we call a treamtment 'treamtment' or 'condition'?

        # get view on treatment region
        treatment_header_idx = sheet.index[sheet[0] == 'Condition'][0]
        treatment_header = sheet.iloc[treatment_header_idx][
            sheet.iloc[treatment_header_idx].notnull()
        ]
        treatment_df = sheet.iloc[
            treatment_header_idx+1:, 0:len(treatment_header)
        ]
        treatment_df.columns = list(treatment_header)
        if 'Round' not in treatment_df:
            treatment_df['Round'] = 'default'
        if 'Fixation Time Point' not in treatment_df:
            treatment_df['Fixation'] = 'default'

        # get view on well region
        well_header = sheet.iloc[0][sheet.iloc[0].notnull()]
        well_index = sheet[0][sheet[0].str.len() == 1]
        well_df = sheet.iloc[well_index.index, well_header.index]
        well_df.columns = map(int, list(well_header))
        well_df.index = list(well_index)
        well_df = well_df.where((pd.notnull(well_df)), None)

        # build condition data frame
        condition_df = pd.DataFrame(
            well_df.values.flatten(),
            columns=["condition"],
            index=[
                "%s%s" % (row_id, column_id)
                for row_id in well_df.index
                for column_id in well_df.columns
            ]
        )
        condition_df.index.name = "well_name"

        # build stain data frame
        conditions = treatment_df['Condition'].unique()
        round_ids = treatment_df['Round'].unique()
        fixation_time_points = treatment_df['Fixation'].unique()
        channel_names = [
            column for column in treatment_df.columns if 'C0' in column
        ]
        channel_ids = [
            column[-1] for column in treatment_df.columns if 'C0' in column
        ]
        stain_df = pd.DataFrame(
            treatment_df[channel_names].values.flatten(),
            columns=["stain_name"],
            index=pd.MultiIndex.from_product(
                [conditions, round_ids, fixation_time_points, channel_ids],
                names=['condition', 'round_id', 'fixation_id', 'channel_id']
            )
        )

        return (condition_df, stain_df)
