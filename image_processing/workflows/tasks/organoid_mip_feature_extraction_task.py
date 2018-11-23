import time
import skimage.measure
import image_processing.feature_extractor

from image_processing.workflows.tasks.base import Task
from image_processing.reader.experiment_reader import *
from image_processing.writer.experiment_writer import TifOvrMipSegFeaturesWriter

class OrganoidMipFeatureExtractionTask(Task):

    extractors = []
    source_path = None

    def __init__(self, extractors=None, weight_channel = None):
        global_config = GlobalConfiguration.get_instance()
        self.source_path = global_config.experiment_default['source_path']
        if extractors is not None:
            extractor_str = extractors
        else:
            extractor_str = global_config.extractors_default['extractors']
        for extractor_lst in extractor_str.split(','):
            extractor_cls = getattr(image_processing.feature_extractor, extractor_lst)
            self.extractors.append(extractor_cls())

        if weight_channel is not None:
            self.weight_channel = weight_channel
        else:
            self.weight_channel = global_config.segmentation_default['weight_channel']

    def _get_region_props(self, label_image, weight_image):

        '''
        Get regionprops weighted by the selected channel
        By default channel
        :param weight_image:
        :return:
        '''

        if not weight_image is None:
            if label_image.size == weight_image.size:
                props = skimage.measure.regionprops(np.asarray(label_image), np.asarray(weight_image), coordinates='rc')
                #props = skimage.measure.regionprops(np.asarray(label_image), np.asarray(weight_image))
        else:
            props = skimage.measure.regionprops(np.asarray(label_image))
        return props

    def run(self, filter = None, append_existing=False, overwrite = False):
        '''

        :param filter: Takes a list of one or multiple plate barcodes as filter arguments ['180823UM1f1'], None processes all plates in experiment
        :return: If true, will try to load existing feature.csv and append new columsn
        TODO --> At the moment if columne already exists a duplicate will be created
        '''
        # Load mips for segmentation
        plates_reader = plateReader()
        tif_image_reader = TifOvrMipReaderPerPlate()
        label_image_reader = TifOvrMipSegReaderPerPlate()
        feature_writer = TifOvrMipSegFeaturesWriter()
        plates_df = plates_reader.read()

        if filter is None:
            filter = list(plates_df['barcode'])
        else:
            # Remove filter values which are not in the experiment folder
            filter = list(set(list(plates_df['barcode'])).intersection(filter))

        plates_to_process = plates_df.loc[plates_df['barcode'].isin(filter)]

        for idx, plate in plates_to_process.iterrows():
            plate_path = os.path.join(plate.plate_path, plate.barcode)
            label_image_df = label_image_reader.read(plate_path)
            if not label_image_df.empty:
                plate_group = pd.concat([label_image_df, tif_image_reader.read(plate_path)])
                segmentation_path = os.path.split(plate_group['file_path'].loc[plate_group['image_type'] == 'label_image'].iloc[0])[0]
                file_name = os.path.join(segmentation_path, 'features', 'features_' + plate.barcode + '.csv')

                well_list = []
                if append_existing & os.path.exists(file_name):
                    # Load existing feature .csv and append
                    feature_df_group = pd.read_csv(file_name).groupby('well')
                else:
                    # Create a new feature list
                    feature_df_group = None

                if not overwrite and os.path.exists(file_name) and not append_existing:
                    print('Featuress for plate: ' + plate.barcode + 'already extracted')
                    continue
                else:
                    for well, well_group in plate_group.groupby('well'):
                        print('Extracting features for plate: ' + plate.barcode + ' well ' + well)
                        if 'label_image' in list(well_group['image_type']):
                            label_image_serie = well_group['image'].loc[well_group['image_type'] == 'label_image']
                            if int(self.weight_channel[-1]) in list(well_group['channel'].loc[well_group['image_type'] == 'intensity_image']):
                                weight_image = well_group['image'].loc[(well_group['image_type'] == 'intensity_image') & (
                                            well_group['channel'] == int(self.weight_channel[-1]))].iloc[0]
                            else:
                                weight_image = None

                            t = time.process_time()
                            regionprops = self._get_region_props(label_image_serie.iloc[0], weight_image)
                            print('\tRegionprops extracted in ' + str(time.process_time() - t))

                            if not feature_df_group is None:
                                feature_df = feature_df_group.get_group(well)
                            else:
                                feature_df = pd.DataFrame()
                            for extractor in self.extractors:
                                feature_df = extractor.extract_features(well_group, feature_df, regionprops)
                            well_list.append(feature_df)
                        else:
                            continue
                    # Collect features for all wells and save
                    if well_list:
                        if len(well_list) > 1:
                            plate_df = pd.concat(well_list, sort=False)
                        else:
                            plate_df = well_list[0]
                    else:
                        plate_df = pd.DataFrame()
                    # Write feature df to disk
                    feature_writer.write(plate_df, file_name)


    def _run(self, filter = None, append_existing=False):
        '''

        :param filter: Takes a list of one or multiple plate barcodes as filter arguments ['180823UM1f1'], None processes all plates in experiment
        :return: If true, will try to load existing feature.csv and append new columsn
        TODO --> At the moment if columne already exists a duplicate will be created
        '''
        # Read in TIF_OVR_MIP and TIF_OVR_MIP_SEG
        mip_reader = TifOvrMipReader()
        seg_reader = TifOvrMipSegReader()
        feature_writer = TifOvrMipSegFeaturesWriter()
        mip_df = mip_reader.read()
        seg_df = seg_reader.read()
        image_df = pd.concat([mip_df, seg_df])

        if filter is None:
            filter = list(seg_df['barcode'])
        else:
            # Remove filter values which are not in the experiment folder
            filter = list(set(list(seg_df['barcode'])).intersection(filter))

        for barcode, plate_group in image_df.groupby('barcode'):
            if barcode in filter:
                plate_path = os.path.split(plate_group['file_path'].loc[plate_group['image_type'] == 'label_image'].iloc[0])[0]
                file_name = os.path.join(plate_path, 'features', 'features_' + barcode + '.csv')
                well_list = []
                if append_existing & os.path.exists(file_name):
                    # Load existing feature .csv and append
                    feature_df_group = pd.read_csv(file_name).groupby('well')
                else:
                    # Create a new feature list
                    feature_df_group = None

                for well, well_group in plate_group.groupby('well'):
                    print('Extracting features for plate: ' + barcode + ' well ' + well)
                    if 'label_image' in list(well_group['image_type']):
                        label_image_serie = well_group['image'].loc[well_group['image_type'] == 'label_image']
                        if int(self.weight_channel[-1]) in list(well_group['channel'].loc[well_group['image_type'] == 'intensity_image']):
                            weight_image = well_group['image'].loc[(well_group['image_type'] == 'intensity_image') & (
                                        well_group['channel'] == int(self.weight_channel[-1]))].iloc[0]
                        else:
                            weight_image = None
                        regionprops = self._get_region_props(label_image_serie.iloc[0], weight_image)
                        if not feature_df_group is None:
                            feature_df = feature_df_group.get_group(well)
                        else:
                            feature_df = pd.DataFrame()
                        for extractor in self.extractors:
                            feature_df = extractor.extract_features(well_group, feature_df, regionprops)
                        well_list.append(feature_df)
                    else:
                        continue
                # Collect features for all wells and save
                if well_list:
                    if len(well_list) > 1:
                        plate_df = pd.concat(well_list)
                    else:
                        plate_df = well_list[0]
                else:
                    plate_df = pd.DataFrame()

                # Write feature df to disk
                feature_writer.write(plate_df, file_name)
            else:
                continue

