import time

from scipy.ndimage import find_objects

from image_processing.workflows.tasks.base import Task
from image_processing.reader.experiment_reader import *
from image_processing.reader.experiment_reader import TifOvrReaderPerPlate, plateReader

class OrganoidTifCroppingTask(Task):

    def __init__(self):
        global_config = GlobalConfiguration.get_instance()
        self.source_path = global_config.experiment_default['source_path']


    def run(self, plate_filter=None, well_filter=None, object_filter=None):
        '''
        Task to crop organoid roi's from stitched z-plane overviews based on bounding boxes extraced from TIF_OVR_MIP_SEG
        '''
        # Load mips for segmentation
        plates_df = plateReader().read()
        tif_image_reader = TifOvrReaderPerPlate()
        label_image_reader = TifOvrMipSegReaderPerPlate()

        if plate_filter is None:
            plate_filter = list(plates_df['barcode'])
        else:
            # Remove filter values which are not in the experiment folder
            plate_filter = list(set(list(plates_df['barcode'])).intersection(plate_filter))

        plates_to_process = plates_df.loc[plates_df['barcode'].isin(plate_filter)]

        for idx, plate in plates_to_process.iterrows():
            plate_path = os.path.join(plate.plate_path, plate.barcode)
            label_image_df = label_image_reader.read(plate_path)
            if not label_image_df.empty:
                plate_group = pd.concat([label_image_df, tif_image_reader.read(plate_path)])
                segmentation_path = os.path.split(plate_group['file_path'].loc[plate_group['image_type'] == 'label_image'].iloc[0])[0]
                crop_path = os.path.join(segmentation_path, 'crop')
                if not os.path.exists(crop_path):
                    os.mkdir(crop_path)

                if well_filter is None:
                    well_filter = list(set(plate_group['well']))
                else:
                    # Remove filter values which are not in the experiment folder
                    well_filter = list(set(set(plate_group['well'])).intersection(well_filter))

                for well, well_group in plate_group.groupby('well'):
                    barcode = well_group.barcode.unique()[0]
                    if well in well_filter:
                        if 'label_image' in list(well_group['image_type']):
                            print('Cropping regions for plate: ' + plate.barcode + ' well ' + well)
                            label_image_serie = well_group['image'].loc[well_group['image_type'] == 'label_image']
                            # TODO --> Give warning if multiple label images are found
                            img_label = label_image_serie.iloc[0]

                            # Generate output path
                            well_crop_path = os.path.join(crop_path, well, 'roi')
                            if not os.path.exists(well_crop_path):
                                os.makedirs(well_crop_path)

                            t = time.process_time()
                            bboxes = find_objects(np.asarray(img_label))
                            tif_image_serie = well_group[['file_name', 'image', 'channel', 'z_stack']].loc[
                                well_group['image_type'] == 'intensity_image']
                            objects = [obj for obj in range(1, len(bboxes) + 1)]
                            if not object_filter is None:
                                # Remove objects values which are not in the experiment folder
                                objects = list(set(set(objects).intersection(object_filter)))

                            for (_, _), group in tif_image_serie.groupby(['channel', 'z_stack']):
                                img_serie = group.iloc[0]
                                for label, bbox in zip(objects, bboxes):
                                    # Crop each region
                                    if label in objects:
                                        try:
                                            img_label_crop = np.asarray(img_label.crop([bbox[1].start, bbox[0].start, bbox[1].stop, bbox[0].stop]))
                                            mask = img_label_crop == label
                                            file_name_mask = 'obj' + str(label) + '_' + barcode + '_' + well + '_mask.tif'
                                            img_crop = img_serie.image.crop([bbox[1].start, bbox[0].start, bbox[1].stop, bbox[0].stop])
                                            file_name = 'obj' + str(label) + '_' + img_serie.file_name
                                            img_crop.save(os.path.join(well_crop_path, file_name), compression="tiff_lzw")

                                            if not os.path.exists(os.path.join(well_crop_path, file_name_mask)):
                                                Image.fromarray(mask.astype(np.uint8)).save(os.path.join(well_crop_path, file_name_mask))
                                        except IOError:
                                            print(file_name + ' could not be written')


