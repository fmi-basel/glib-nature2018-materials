from reader.experiment_reader import TifOvrMipReader
from utils.global_configuration import GlobalConfiguration
import os
import re
import numpy as np
from PIL import Image
from skimage.external.tifffile import imread as tiffread
import pandas as pd

class RGBOverviewCreationTask:

    def __init__(self, source_path = None):
        if source_path is not None:
            self.source_path = source_path
        else:
            global_config = GlobalConfiguration.get_instance()
            self.source_path = global_config.experiment_default['source_path']

    def run(self,
            rgb_channel_composition = '3,2,1',
            rgb_channel_clipping = "(200,3000),(50,800),(50,1000)",
            rgb_overview_shrinkage = '3x',
            shrinkage_to_load ='1x'):

        """
        Main method.
        """

        # disable DecompressionBombError while open large image data
        Image.MAX_IMAGE_PIXELS = np.inf
        self.shrinkage = rgb_overview_shrinkage

        if rgb_channel_composition != "auto":
            red_composition, green_composition, blue_composition = (
                (rgb_channel_composition.split(",") + [None] * 3)[0:3]
            )

            if not any([red_composition, green_composition, blue_composition]):
                raise RuntimeError("No valid channels defined to compose!")
            for channel_name, channel_composition in [
                ("red", red_composition),
                ("green", green_composition),
                ("blue", blue_composition)
            ]:
                if (
                        channel_composition is not None and
                        not re.match(r"^\d?([+-]\d)*$", channel_composition)
                ):
                    raise RuntimeError(
                        "Invalid defined '%s' for channel '%s'!"
                        % (channel_composition, channel_name)
                    )

        # Save composition
        comp = rgb_channel_composition.split(',')
        clip_temp = rgb_channel_clipping.replace('(', '').split(')')
        clip_temp = [c.lstrip(',').split(',') for c in clip_temp]
        color_dict = {0: 'red', 1: 'green', 2: 'blue'}
        rgb_meta = []
        for idx, (ch, clip) in enumerate(zip(comp, clip_temp)):
            rgb_meta.append(
                {'color': color_dict[idx], 'channel': 'c' + ch, 'lower_clipping': clip[0],
                 'upper_clipping': clip[1]})
        meta_df = pd.DataFrame(rgb_meta)

        # gather data
        if self.shrinkage == shrinkage_to_load:
            # Load pre-shrinked data to speed up
            reader = TifOvrMipReader(shrinkage= shrinkage_to_load)
            mip_overview = reader.read()
        else:
            reader = TifOvrMipReader()
            mip_overview = reader.read()

        for (barcode, well), well_group in mip_overview.groupby(['barcode', 'well']):
            print(barcode, well)

            # create RGB directory
            base_path = well_group['file_path'].unique()[0]
            if 'shrinkage' in base_path:
                base_path = os.path.dirname(base_path)
            rgb_overview_path = base_path.replace('TIF_OVR_MIP','RGB_OVR_MIP')

            if not os.path.exists(rgb_overview_path):
                os.makedirs(rgb_overview_path)
            meta_df.to_csv(os.path.join(rgb_overview_path, 'channel_order.csv'))

            try:
                (rgb_overview_width, rgb_overview_height) = well_group.iloc[0]['image'].size
                # temporarily store the rgb overview as int32 image.
                # hereby, we can support '-' and '+' operations which
                # may result in negative values or values greater then 16bit.
                rgb_overview = np.zeros(
                    (rgb_overview_height, rgb_overview_width, 3), np.int32
                )

                rgb_overview_name = re.sub(
                    "C\d{2}.tif", "RGB.tif", well_group.iloc[0]["file_name"]
                )

                channel_index = {}
                for row_idx, file_row in well_group.iterrows():
                    channel_id = file_row["channel"]
                    if channel_id in channel_index:
                        print(
                            "Found multiple images with channel id '%s' "
                        ) % (channel_id)
                        continue
                    channel_index[channel_id] = row_idx

                if rgb_channel_composition == "auto":
                    red_composition, green_composition, blue_composition = (
                        (sorted(channel_index.keys()) + [None] * 3)[0:3]
                    )

                for channel_idx, channel_composition in enumerate([
                    red_composition, green_composition, blue_composition
                ]):
                    if channel_composition is None:
                        continue

                    for pos in range(0, len(channel_composition), 2):
                        channel_id = channel_composition[pos]
                        file_row = well_group.loc[channel_index[int(channel_id)]]
                        file_name = file_row["file_name"]
                        file_path = os.path.join(
                            file_row["file_path"], file_name
                        )

                        ch_image = tiffread(file_path)# Only read first layer of tiff
                        # TODO --> At the moment only works with shrinked folders, prolbem to loas tiff with two layers
                        if pos - 1 < 0:
                            rgb_overview[:, :, channel_idx] = ch_image
                        elif channel_composition[pos - 1] == '+':
                            rgb_overview[:, :, channel_idx] += ch_image
                        elif channel_composition[pos - 1] == '-':
                            rgb_overview[:, :, channel_idx] -= ch_image
                        else:
                            raise RuntimeError(
                                (
                                    "'%s' is not a supported channel "
                                    "composition operator!"
                                ) % channel_composition[pos - 1]
                            )

                        # free memory
                        del ch_image

                # apply clipping
                if rgb_channel_clipping != "skip":
                    try:
                        clipping_ranges = (
                            re.search(
                                "\((\d+,\d+)\),\((\d+,\d+)\),\((\d+,\d+)\)",
                                rgb_channel_clipping
                            ).groups()
                        )
                        clipping_ranges = map(
                            lambda clipping_range: map(
                                int, clipping_range.split(',')
                            ),
                            clipping_ranges
                        )

                        for cn_idx, clipping_range in enumerate(
                                clipping_ranges
                        ):
                            rgb_overview[:, :, cn_idx] = (
                                rgb_overview[:, :, cn_idx].clip(
                                    *clipping_range
                                )
                            )
                    except Exception as ex:
                        rgb_channel_clipping = "skip"

                # apply linear contrast stretching
                """
                min_rgb_intensity = np.min(rgb_overview)
                max_rgb_intensity = np.max(rgb_overview)
                if min_rgb_intensity != max_rgb_intensity:
                    for cn_idx in range(3):
                        cn_image = rgb_overview[:, :, cn_idx]
                        rgb_overview[:, :, cn_idx] = (
                            (cn_image-min_rgb_intensity) /
                            (max_rgb_intensity-min_rgb_intensity) * 255
                        )
                """
                for cn_idx in range(3):
                    cn_image = rgb_overview[:, :, cn_idx]
                    if np.min(cn_image) == np.max(cn_image):
                        continue

                    rgb_overview[:, :, cn_idx] = (
                            (cn_image - np.min(cn_image)) /
                            (np.max(cn_image) - np.min(cn_image)) * 255
                    )

                rgb_overview = rgb_overview.astype(np.uint8)

                # binning the rgb_overview. adapted from
                # https://scipython.com/blog/binning-a-2d-array-in-numpy
                # images which do not fit the bin_shape are cut accordingly!
                if self.shrinkage != shrinkage_to_load:
                    bin_shape = (
                        rgb_overview_height // rgb_overview_shrinkage,
                        rgb_overview_shrinkage,
                        rgb_overview_width // rgb_overview_shrinkage,
                        rgb_overview_shrinkage
                    )
                    scaled_rgb_overview = np.zeros(
                        (
                            rgb_overview_height // rgb_overview_shrinkage,
                            rgb_overview_width // rgb_overview_shrinkage,
                            3
                        ), np.uint8
                    )
                    round_cn_image_height = rgb_overview_height - (
                            rgb_overview_height % rgb_overview_shrinkage
                    )
                    round_cn_image_width = rgb_overview_width - (
                            rgb_overview_width % rgb_overview_shrinkage
                    )
                    for cn_idx in range(3):
                        cn_image = rgb_overview[
                                   0:round_cn_image_height, 0:round_cn_image_width, cn_idx
                                   ]
                        scaled_rgb_overview[:, :, cn_idx] = (
                            cn_image.reshape(bin_shape).mean(-1).mean(1)
                        )
                else:
                    scaled_rgb_overview = rgb_overview


                Image.fromarray(scaled_rgb_overview).save(
                    os.path.join(rgb_overview_path, rgb_overview_name),
                    compression="tiff_lzw"
                )
                # free memory
                del rgb_overview

            except:
                pass

