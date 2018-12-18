import os
import logging

import numpy as np
from tqdm import tqdm

from skimage.external.tifffile import imread, imsave
from skimage.segmentation import watershed

from dlutils.postprocessing.watershed import segment_nuclei
from image_processing.reader.stackreader import StackGenerator


class NucleusAndCytosolWatershed(object):

    input_channel_pattern = {'CellTrace': '*C04.tif', 'DAPI': '*C01.tif'}

    @staticmethod
    def get_additional_stack(dirname, paths):
        '''
        '''
        additional_stack = [
            imread(os.path.join(dirname, os.path.basename(path)))
            for path in sorted(paths)
        ]
        return np.asarray(additional_stack).squeeze()

    def segment(self, stack_dir, out_dir):
        '''applies a watershed segmentation to the probability maps in the given
        stack_dir.

        :param stack_dir: path to stack
        :param out_dir: path to save outputs

        '''

        # load stacks
        logging.getLogger(__name__).info('Processing folder: %s', stack_dir)

        if os.path.abspath(stack_dir) == os.path.abspath(out_dir):
            raise ValueError('Input and output directory cannot be the same!')

        for stack_cp, stack_org in self.generator_fn(stack_dir, out_dir):
            pred_stack = self.processor_fn(
                cell_segm=stack_cp['segm_stack'],
                separator=stack_cp['separator_stack'],
                org_segm=stack_org['segm_stack'])

            self.save_fn(
                stack=stack_cp, pred_stack=pred_stack, out_dir=out_dir)

        logging.getLogger(__name__).info('Done')

    def generator_fn(self, stack_dir, pred_dir):
        '''reads input stacks.

        '''

        generator_a = StackGenerator(
            img_dir=stack_dir,
            segm_dir=os.path.join(pred_dir, 'cell_pred'),
            pattern=[self.input_channel_pattern['DAPI'], '*mask.tif'])

        generator_b = StackGenerator(
            img_dir=stack_dir,
            segm_dir=pred_dir,
            pattern=[self.input_channel_pattern['CellTrace'], '*mask.tif'])

        assert len(generator_a) == len(generator_b)

        for stack_cp, stack_org in tqdm(
                zip(generator_a, generator_b), leave=True, ncols=80,
                desc='Processing well', total=len(generator_a)):

            stack_cp['separator_stack'] = self.get_additional_stack(
                os.path.join(pred_dir, 'border_pred'), stack_cp['image_paths'])

            # rescale probabilities
            stack_cp['segm_stack'] = stack_cp['segm_stack'] / 255.
            stack_cp['separator_stack'] = stack_cp['separator_stack'] / 255.
            stack_org['segm_stack'] = stack_org['segm_stack'] / 255.

            assert stack_cp['separator_stack'].shape == stack_cp[
                'segm_stack'].shape

            yield stack_cp, stack_org

    def processor_fn(self, cell_segm, separator, org_segm):
        '''apply watershed to all input probability maps.

        '''
        assert cell_segm.shape == separator.shape
        assert cell_segm.shape == org_segm.shape

        pred_stack = {'cytosol_segm': [], 'nuclei_segm': []}

        for idx in range(len(cell_segm)):

            # watershed on nucleus and separator probs.
            nuclei_segm = segment_nuclei(
                cell_segm[idx],
                separator[idx],
                threshold=0.5,
                upper_threshold=0.8,
            )

            # Assign cytosol to nearest nucleus and
            # remove cytosol where there is a nucleus.
            fg_mask = org_segm[idx] >= 0.5
            fg_segm = watershed(
                1 - org_segm[idx], markers=nuclei_segm, mask=fg_mask)
            fg_segm[nuclei_segm >= 1] = 0

            pred_stack['cytosol_segm'].append(fg_segm)
            pred_stack['nuclei_segm'].append(nuclei_segm)

        return pred_stack

    @staticmethod
    def get_outpath(out_dir, sub_dir, orig_path):
        '''create output path from output base directory and original name.

        '''
        dirname = os.path.join(out_dir, sub_dir)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            logging.getLogger(__name__).debug('Creating output folder at %s',
                                              dirname)
        return os.path.join(dirname, os.path.basename(orig_path))

    def save_fn(self, stack, pred_stack, out_dir):
        '''save output masks.

        '''
        for key in pred_stack.keys():
            dirname = os.path.join(out_dir, key)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                logging.getLogger(__name__).debug(
                    'Creating output folder at %s', dirname)

        assert len(stack['image_paths']) == len(pred_stack['cytosol_segm'])
        assert len(stack['image_paths']) == len(pred_stack['nuclei_segm'])

        for idx, orig_path in enumerate(stack['image_paths']):
            for key, val in pred_stack.items():
                imsave(
                    self.get_outpath(out_dir, key, orig_path),
                    val[idx].astype(np.uint16))
