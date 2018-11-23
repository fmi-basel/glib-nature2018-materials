import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_dilation

from feature_extractor.utils.focus import predict_infocus
from reader.stackreader import StackGenerator
from features.features import segment, calc_base_features, calc_derived_features


def segment(prob, image, threshold, focus_threshold, max_component=False):
    '''generates binary mask from probability map and image stack.

    Parameters
    ----------
    prob : array_like, shape=(Z, X, Y)
        Probability map.
    image : array_like, shape=(Z, X, Y)
        Corresponding image stack.
    threshold : float, [0., 1.]
        Threshold on probability map.
    focus_threshold : float, [0, 1]
        Threshold on in-focus probability. Planes with probabilities
        lower than focus_threshold are not considered for segmentation.
    max_component : bool, default=False
        Consider only largest component of segmentation.

    Returns
    -------
    mask : array_like, shape=(Z, X, Y)
        binary segmentation mask.
    focus_prob : array_like, shape=(Z,)
        probabilities of in-focus model for each plane.

    '''
    focus_prob = predict_infocus(image)

    mask = prob >= threshold * 255
    mask[focus_prob < focus_threshold] = 0

    if max_component:
        connected_components, n_components = label(mask)
        if n_components <= 1:
            return mask

        sizes = [(connected_components == ii).sum()
                 for ii in range(1, n_components)]
        idx = np.argmax(sizes) + 1
        mask = connected_components == idx
    return mask, focus_prob


def segment_lumen(segm):
    '''plane-wise filling only.

    '''
    filled = np.asarray([binary_fill_holes(plane) for plane in segm])
    return np.logical_and(np.logical_not(segm), filled)


def coords_from_segm(segm, aspect, resample=True):
    '''generate coordinates from segmentation. Z-axis is resampled w.r.t.
    min/max aspect ratio.

    '''
    assert segm.ndim == len(aspect)

    if resample:
        axis = np.argmax(aspect)
        kernel = np.ones(tuple(2 if ii == axis else 1 for ii in range(segm.ndim)))
        segm = binary_dilation(segm, kernel)

    coords = np.vstack(np.where(segm)).T.astype(float)
    coords *= aspect

    if resample:
        coords[:, axis] -= max(aspect) / 2.

    return coords


def sparse_inertia_tensor(coords):
    '''calculate inertia tensor from coordinates.

    '''
    assert coords.ndim == 2
    assert coords.shape[-1] == 3

    coords_centralized = coords - coords.mean(axis=0)
    # the - and successive for loop is due to convention for inertia tensors.
    inertia = -np.dot(coords_centralized.T, coords_centralized) / len(coords)
    for ii in range(3):
        inertia[ii, ii] *= -1
    return inertia


def convex_hull_volume(coords):
    '''
    '''
    assert coords.ndim == 2
    assert coords.shape[-1] == 3
    hull = ConvexHull(coords)
    return hull.volume


def calc_base_features(segm, mask, spacing):
    '''calculate basic features from segmented stack.

    '''
    features = {}

    assert segm.ndim == len(spacing)

    try:
        features['volume'] = segm.sum() * np.prod(spacing)
        features['lumen'] = segment_lumen(segm).sum() * np.prod(spacing)
        features['mask_area'] = mask.sum() * np.prod(spacing[1:])
        features['projection_area'] = segm.max(axis=0).sum() * np.prod(
            spacing[1:])

        coords = coords_from_segm(segm, aspect=spacing, resample=True)
        features['convex_hull_volume'] = convex_hull_volume(coords)

        tensor = sparse_inertia_tensor(coords)

        eigenvals, eigenvecs = np.linalg.eigh(tensor)
        # reverse eigenvalues to have them in descending order
        eigenvals = eigenvals[::-1]
        eigenvecs = eigenvecs[:, ::-1]

        features['eccentricity'] = np.sqrt(1 - eigenvals[1] / eigenvals[0])

        # minor and major axis length as in skimage.measure.regionprobs
        features['minor_axis_length'] = 4 * np.sqrt(eigenvals[1])
        features['major_axis_length'] = 4 * np.sqrt(eigenvals[0])
        for ii, eigenval in enumerate(eigenvals):
            features['eigenval_{}'.format(ii)] = eigenval

        features['elevation_angle_deg'] = np.arcsin(
            eigenvecs[0, 0]) / np.pi * 180

    except QhullError:
        pass
    except ValueError as err:
        if 'No points given' in str(err):
            pass
        else:
            raise
    return features


def calc_derived_features(df):
    '''
    '''
    df = df.assign(solidity=df['volume'] / df['convex_hull_volume'])
    df = df.assign(axis_ratio=df['major_axis_length'] /
                   df['minor_axis_length'])
    df = df.assign(mask_to_projection_ratio=df['projection_area'] /
                   df['mask_area'])
    return df
