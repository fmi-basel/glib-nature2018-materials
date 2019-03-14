import abc, six
import os
import imghdr
import re
from PIL import Image

@six.add_metaclass(abc.ABCMeta)
class Reader():
    @abc.abstractmethod
    def read(self, source_path, parameter):
        pass

    def _get_images(self, file_path, image_list, type, shrinkage=None, segmentation_tag=None):
        files = [file for file in os.listdir(file_path) if
                 os.path.isfile(os.path.join(file_path, file))]

        file_reg = re.compile(r'\w.*_\w{3}_T\w{4}F\w{3}L\w{2,3}A\w{2}Z\w{2}')
        if type == 'rgb_image':
            meta_reg = re.compile(r'_\w{3}_T\w{4}F\w{3}L\w{2,3}A\w{2}Z\w{2}RGB.')
        else:
            meta_reg = re.compile(r'_\w{3}_T\w{4}F\w{3}L\w{2,3}A\w{2}Z\w{2}C\w{2}.')


        barcode_reg = re.compile("\d{6}\w{2}.{3,6}_")
        well_reg = re.compile(r'^\w{1}\d{2}_')
        time_reg = re.compile(r'_T\d{4}F')
        field_reg = re.compile(r'F\d{3}L')
        line_reg = re.compile(r'L\d{2}A')
        action_reg = re.compile(r'A\d{2}Z')
        z_stack_reg = re.compile(r'Z\d{2}C')
        channel_reg = re.compile(r'C\d{2}\.')

        for file in [f for f in files if file_reg.match(f)]:
            # If files is an image extract well and store
            if imghdr.what(os.path.join(file_path, file)):
                # Exract image informations
                meta = meta_reg.search(file).group(0).strip('_')
                barcode = barcode_reg.search(file).group(0).strip('_')
                well = well_reg.search(meta).group(0).strip('_')
                time_point = int(time_reg.search(meta).group(0).strip('F').strip('_').strip('T').lstrip('0'))
                line = int(line_reg.search(meta).group(0).strip('L').strip('A').lstrip('0'))
                action = int(action_reg.search(meta).group(0).strip('A').strip('Z').lstrip('0'))
                field = int(field_reg.search(meta).group(0).strip('F').strip('L').lstrip('0'))
                if type == 'rgb_image':
                    z_stack = None
                    channel = None
                else:
                    z_stack = int(z_stack_reg.search(meta).group(0).strip('Z').strip('C').lstrip('0'))
                    channel = int(channel_reg.search(meta).group(0).strip('C').strip('.').lstrip('0'))
                image = Image.open(os.path.join(file_path, file))

                image_list.append(
                    {'file_path': file_path,
                     'file_name': file,
                     'image_type': type,
                     'segmentation_tag': segmentation_tag,
                     'shrinkage': shrinkage,
                     'barcode': barcode,
                     'well': well[0] + well[1:].lstrip('0'),
                     'time_point': time_point,
                     'line': line,
                     'action': action,
                     'field': field,
                     'z_stack': z_stack,
                     'channel': channel,
                     'image': image})

        return image_list


    def __get_images(self, file_path, image_list, type, shrinkage=None, segmentation_tag=None):
        files = [file for file in os.listdir(file_path) if
                 os.path.isfile(os.path.join(file_path, file))]

        meta_reg = re.compile(r'_\w{3}_T\w{4}F\w{3}L\w{2,3}A\w{2}Z\w{2}C\w{2}.')
        barcode_reg = re.compile("\d{6}\w{2}.{3,6}_")
        well_reg = re.compile(r'^\w{1}\d{2}_')
        time_reg = re.compile(r'_T\d{4}F')
        field_reg = re.compile(r'F\d{3}L')
        line_reg = re.compile(r'L\d{2}A')
        action_reg = re.compile(r'A\d{2}Z')
        z_stack_reg = re.compile(r'Z\d{2}C')
        channel_reg = re.compile(r'C\d{2}\.')

        # Only keep valid images and extract image type
        for file in files:
            # If files is an image extract well and store
            if imghdr.what(os.path.join(file_path, file)):
                # Exract image informations
                meta = meta_reg.search(file).group(0).strip('_')
                barcode = barcode_reg.search(file).group(0).strip('_')
                well = well_reg.search(meta).group(0).strip('_')
                time_point = int(time_reg.search(meta).group(0).strip('F').strip('_').strip('T').lstrip('0'))
                line = int(line_reg.search(meta).group(0).strip('L').strip('A').lstrip('0'))
                action = int(action_reg.search(meta).group(0).strip('A').strip('Z').lstrip('0'))
                field = int(field_reg.search(meta).group(0).strip('F').strip('L').lstrip('0'))
                z_stack = int(z_stack_reg.search(meta).group(0).strip('Z').strip('C').lstrip('0'))
                channel = int(channel_reg.search(meta).group(0).strip('C').strip('.').lstrip('0'))
                image = Image.open(os.path.join(file_path, file))

                image_list.append(
                    {'file_path': file_path,
                     'file_name': file,
                     'image_type': type,
                     'segmentation_tag': segmentation_tag,
                     'shrinkage': shrinkage,
                     'barcode': barcode,
                     'well': well[0] + well[1:].lstrip('0'),
                     'time_point': time_point,
                     'line': line,
                     'action': action,
                     'field': field,
                     'z_stack': z_stack,
                     'channel': channel,
                     'image': image})

        return image_list
