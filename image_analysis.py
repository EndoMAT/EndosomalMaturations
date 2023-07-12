#!/usr/bin/env python

import os
import sys

from matplotlib import pyplot as plt
import numpy as np

from functools import wraps
from scipy import cluster, interpolate, ndimage, signal, spatial, stats
from skimage import exposure, feature, filters, measure, morphology, segmentation
from skimage.util import crop, invert, img_as_uint, img_as_float
from sklearn.cluster import KMeans

from data_collections import *


def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            return func(*args, **kwargs)
    return wrapper


def check_arr_shape(func):
    @wraps(func)
    def inner(ndarray, *args, **kwargs):
        # Get array shape, assuming first argument
        shape = np.asarray(ndarray.shape)
        
        # Find the axes with length = 1
        axis = tuple(np.flatnonzero(shape == 1))
        reshape = len(axis) > 0
        
        # Remove axes with length = 1
        if reshape:
            ndarray = ndarray.squeeze()
        
        # Calculate the result
        result = func(ndarray, *args, **kwargs)
        
        # Expand the result to match the input dimensions
        if reshape:
            result = np.expand_dims(result, axis=axis)
        
        #  Return the final result
        return result
        
    return inner


def get_labels(labeled, bg_val=0):
    """ List all labels in input labeled image.
    """

    # Get unique values in input image
    labels = np.unique(labeled)
    
    # Remove objects corresponding to background
    labels = labels[labels != bg_val]
    
    return sorted(labels)


def pad_overhang(original):
    """ Remove overhang in deskewed image by tiling.
    """
        
    # Extract regions with blank values and dilate to ensure no zeros remain
    zero_mask = ndimage.binary_dilation(original == 0, iterations=5)

    # Label the objects in the mask
    zero_labeled, _ = ndimage.label(zero_mask)
    zero_labels = get_labels(zero_labeled)

    # Extract the bounding box around each zeros region
    zero_slices = ndimage.find_objects(zero_labeled)
    zero_rois = [original[zero_slice] for zero_slice in zero_slices]

    rectangular = np.copy(original)
    for zero_slice in zero_slices:
        # Extract the region containing zero values
        zero_roi = original[zero_slice]

        # Flip and rotate the blank region (assumes identical deskew procedure)
        flipped_roi = np.transpose(np.rot90(np.transpose(np.rot90(zero_roi, 2)), 2))

        # Fill all zeros from the flipped region
        zero_roi_mask = zero_roi == 0
        rectangular[zero_slice][zero_roi_mask] = flipped_roi[zero_roi_mask]
    
    return rectangular


@check_arr_shape
def filter_image(image_in, kernel_size=None, size=None, footprint=None, gamma=None, h_level=None):
    """ Filter image, seeking to identify blobs.
    """
    
    # Rectangularize the deskewed image
    image_out = pad_overhang(image_in)
    
    # Apply median filter
    image_out = ndimage.median_filter(image_out, size=size, footprint=footprint)
    
    # Equalize input image using adaptive histogram
    image_out = exposure.rescale_intensity(image_out, out_range=(0, 1))
    image_out = exposure.equalize_adapthist(image_out, kernel_size=kernel_size)
    
    return img_as_uint(image_out)

def locate_blobs(dset_in, *args, param_dict=dict(), method='log', **kwargs):
    """ Locate centers of blobs within image.
    """
    
    # Set parameters to convert into DataFrame
    columns = 'z_px', 'y_px', 'x_px', 'sigma_z', 'sigma_y', 'sigma_x'
    type_mapper = {'z_px' : int, 'y_px' : int, 'x_px' : int}
    
    # Read channels from dataset variables
    channels = list(dset_in.data_vars)
    
    if method == 'log':
        # Find blobs using Laplacian of Gaussian
        func = feature.blob_log
    elif method == 'dog':
        # Find blobs using Differences of Gaussian
        func = feature.blob_dog
    
    dfs = dict.fromkeys(channels)
    for channel in channels:
        image = dset_in[channel].data
        in_range = np.nanmin(image), np.nanmax(image)
        image = exposure.rescale_intensity(image, in_range=in_range, out_range=(0, 1))
        
        # Incorporate channel-specific and -general parameters
        if param_dict is not None:
            params = {**param_dict[channel], **kwargs}
        else:
            params = kwargs
        
        # Calculate blobs for each channel
        data = func(image, **params)
        
        # Convert results to DataFrame
        df = pd.DataFrame(data=data, columns=columns)
            
        # Change dtype of coordinates
        df = df.astype(type_mapper)
        
        dfs[channel] = df
    
    return dfs


def fill_blank_regions(image, cval=0):
    """ Use distance transform to fill holes in input image.
    """
    
    # Get mask showing locations of zeros
    mask_zeros = image == cval

    # Perform distance transform to fill holes in image
    distances = ndimage.distance_transform_edt(mask_zeros, return_distances=False, return_indices=True)
    fill_values = image[tuple(distances)]

    # Replace zero-valued pixels from image2
    padded_image = np.copy(image)
    padded_image[mask_zeros] = fill_values[mask_zeros]

    return padded_image

def remove_objects(image, mask, footprint=morphology.ball(3), iterations1=3, iterations2=None, max_quantile=0.5, min_value=-1):
    """ Fill bright objects in image, approximating local background.
    """
    
    # Check values of expand2
    if iterations2 is None:
        iterations2 = iterations1 + 1
    if iterations2 <= iterations1:
        raise ValueError("Input iterations2 must be greater than iterations1.")
    
    # Calculate dilated masks over blobs (bright objects in image)
    dilated_mask = ndimage.binary_dilation(mask, footprint, iterations=iterations1)
    
    # Calculate based on difference in masks
    shell_mask = ndimage.binary_dilation(mask, footprint, iterations=iterations2)
    shell_mask[dilated_mask] = False
    
    # Create image with holes to be filled
    zeroed = np.copy(image)
    zeroed[dilated_mask] = min_value
    
    # Remove any other pixels of sufficiently high intensity
    cutoff = np.quantile(zeroed[shell_mask], max_quantile)
    zeroed[shell_mask > cutoff] = min_value
    
    # Fill holes in image
    filled = fill_blank_regions(zeroed, min_value)
    
    return filled

def approximate_size(ndim=3, radius=None):
    """ Approximate object size for bulk intensity calculation around point without segmentation.
    """
    
    if radius is None:
        # ndim = 2 : morphology.disk(2).sum() = 13
        # ndim = 3 : morphology.ball(2).sum() = 33
        size = {2 : 13, 3 : 33}[ndim]
    else:
        func = {2 : morphology.disk, 3 : morphology.ball}[ndim]
        size = func(radius).sum()
    
    return size
    
def approximate_signal(arr, quantile=0.9, radius=None, func=np.median):
    """ Approximate signal as average of brightest pixels.
    """
    
    min_size = approximate_size(ndim=3, radius=radius)
    
    if arr.size < min_size:
        top = arr
    else:
        if quantile is None:
            top = np.sort(arr)[-min_size:]
        else:
            top = arr[arr >= np.quantile(arr, quantile)]
    
    signal = func(top)
    
    return signal

def approximate_background(arr, quantile=0.1, radius=None, func=np.median):
    """ Approximate background as average of dimmest pixels.
    """
    
    min_size = approximate_size(ndim=3, radius=radius)
    
    if arr.size < min_size:
        bottom = arr
    else:
        if quantile is None:
            bottom = np.sort(arr)[:min_size]
        else:
            bottom = arr[arr <= np.quantile(arr, quantile)]
    
    signal = func(bottom)
    
    return signal


def split_blobs(blobs, ndim=3):
    """ Separate coordinates from sigma values in blobs array.
    """
    
    # Separate coordinates from sigma values
    coords = np.array(blobs[:, :ndim], dtype=int)
    sigmas = blobs[:, ndim:]
    
    return coords, sigmas


def filter_blobs(blobs, mask, min_sigma=None, max_sigma=None, exclude_border=False):
    """ Filter out blobs based on location and size.
    """
    
    # Read image data
    ndim = mask.ndim
    shape = np.asarray(mask.shape)
    
    # Split into coordinates and sigma values
    coords, sigmas = split_blobs(blobs, ndim=ndim)
    
    # Generate indexes of blobs to retain
    indexes = set(range(len(blobs)))
    
    for index, (coord, sigma) in enumerate(zip(coords, sigmas)):
        keep = True
        
        # Check that blobs centers are located in mask
        if not mask[tuple(coord)]:
            keep = False
        # Check that blob has sigma greater than min value
        elif min_sigma is not None and np.any(sigma < min_sigma):
            keep = False
        # Check that blob has sigma less than max value
        elif max_sigma is not None and np.any(sigma > max_sigma):
            keep = False
        # Check whether center is within specified distance of border
        elif exclude_border:
            min_coord = exclude_border
            max_coord = shape - exclude_border
            if any(coord < min_coord) or any(coord > max_coord):
                keep = False
        
        np.asarray(mask.shape)
        # Discard blob if any of above checks not met
        if not keep:
            indexes.discard(index)
    
    return sorted(indexes)


def labeled_comprehension(image, labeled, func, labels=None, out_dtype=float, default=0):
    """ Apply function to labeled objects.
    
        image: data array
        labeled: labeled array
        func: arbitrary function
    """
    
    if labels is None:
        labels = get_labels(labeled)
    
    # Apply function to all labeled objects
    result = ndimage.labeled_comprehension(image, labeled, labels, func, out_dtype, default)
    
    return result


def sort_labeled_objects(image, labeled, func, labels=None, out_dtype=None, default=0):
    """ Apply function to labeled objects then return labels sorted accordingly.
    
        image: data array
        labeled: labeled array
        func: arbitrary function
    """
    
    if labels is None:
        labels = get_labels(labeled)
    
    # Apply function to all labeled objects
    result = labeled_comprehension(image, labeled, func, labels, out_dtype, default)
    
    # Order objects based on size
    indexes = np.argsort(result)
    
    # Sort labels accordingly
    sorted_labels = [labels[index] for index in indexes]
    
    return sorted_labels


def find_roi(coord, radius, shape):
    # Get image dimensions
    ndim = len(coord)
    axes = range(ndim)
    
    # Generate mask as bounding box
    slices = []
    for axis in axes:
        # Lower and upper values must be within image
        lower = max(coord[axis] - radius[axis], 0)
        upper = min(coord[axis] + radius[axis], shape[axis])
        slices.append(slice(lower, upper + 1))
    
    return tuple(slices)


def extract_blob(image, coord, radius):
    slices = find_roi(coord, radius, image.shape)
    return image[slices]


def get_radii(sigmas, expand=0, rounding='up', out_dtype=int):
    # Set number of dimensions based on input
    ndim = sigmas.shape[1]
    
    # Convert from sigmas to radii
    radii = np.sqrt(ndim) * sigmas
    
    # Dilate or erode radii based on value of expand
    radii += expand
    
    # Do not allow complete erosion (radii cannot be negative)
    if expand < 0: radii = np.clip(radii, 0, None)
    
    # Round to nearest integer if rounding specified
    if rounding == 'down': radii = np.floor(radii)
    elif rounding == 'up': radii = np.ceil(radii)
    
    # Convert to request data type
    radii = radii.astype(out_dtype)
    
    return radii


def mask_coords(coords, mask=None, shape=None):
    if mask is None: mask = np.zeros(shape, dtype=bool)
    mask[tuple(zip(*coords))] = True
    return mask


def label_coords(coords, mask=None, shape=None):
    mask = mask_coords(coords, mask=mask, shape=shape)
    labeled, _ = ndimage.label(mask)
    return labeled


def get_footprint(radius, sizes, rounded=True, dtype=bool):
    # Ensure that relative sizes are normalized to steps
    steps = (np.array(sizes) / np.min(sizes)).astype(int)
    
    if rounded:
        # Generate ball structuring element for largest dimension
        footprint = morphology.ball(radius, dtype=dtype)
        input_shape = footprint.shape
        
        # Remove slices if relative sizes not all equal
        output_slices = []
        for input_len, step in zip(input_shape, steps):
            start = (input_len // 2) % step if step > 1 else 0
            output_slices.append(slice(start, None, step))
        
        # Slice original footprint
        footprint = footprint[tuple(output_slices)]
    else:
        # Convert from single value of radius based on relative step sizes
        output_shape = tuple(2 * (radius // steps) + 1)
        
        # Rectangular matrix of ones
        footprint = np.ones(output_shape, dtype=dtype)
    
    return footprint


def get_footprint_coords(footprint):
    where = np.where(footprint)
    shape = footprint.shape
    args = zip(where, shape)
    coords = [arg[0] - (arg[1] // 2) for arg in args]
    return list(zip(*coords))


def get_blob_coords(blob_coord, footprint_coords, output_shape):
    # Center footprint on blob
    coords = blob_coord + footprint_coords
    
    # Remove coordinates that fall outside the image
    coords = coords[np.all(coords >= 0, axis=1)]
    coords = coords[np.all(coords < output_shape, axis=1)]
    
    return coords


def mask_blobs(blobs, shape, sizes=(2, 1, 1)):
    # Get coordinates and sigmas from blobs matrix
    coords, sigmas = split_blobs(blobs)
    
    # Get maximum radius in each dimension
    radii = np.max(get_radii(sigmas), axis=1)
    
    # Make footprints dictionary
    footprint_coords_dict = dict()
    for radius in np.unique(radii):
        footprint = get_footprint(radius, sizes)
        footprint_coords = get_footprint_coords(footprint)
        footprint_coords_dict[radius] = footprint_coords
    
    blobs_mask = np.zeros(shape, dtype=bool)
    for coord, radius in zip(coords, radii):
        footprint_coords = footprint_coords_dict[radius]
        blob_coords = get_blob_coords(coord, footprint_coords, shape)
        blobs_mask = mask_coords(blob_coords, mask=blobs_mask)
    
    return blobs_mask


def clip_array(array, proportion=0.1, sigma=None):
    if sigma:
        clipped, _, _ = stats.sigmaclip(array, low=sigma, high=sigma)
    else:
        clipped = stats.trimboth(array, proportion, axis=None)
    return clipped


def trim_mean(array, *args, **kwargs):
    clipped = clip_array(array, *args, **kwargs)
    return np.mean(clipped)


def sigmaclip_mean(array, sigma=1.0):
    return trim_mean(array, sigma=1.0)


def interp_missing(invalid_arr, method='linear'):
    # Mask invalid values
    nan_mask = np.ma.masked_invalid(invalid_arr).mask
    
    # Create grid data for indices
    coords = [np.arange(0, n) for n in invalid_arr.shape]
    grids = tuple(np.meshgrid(*coords, indexing='ij'))
    
    # Get points and values
    points = np.transpose([grid[~nan_mask] for grid in grids])
    values = invalid_arr[~nan_mask].ravel()
    
    # Fill points using specified method ('linear' or 'nearest')
    filled_arr = interpolate.griddata(points, values, grids, method=method)
    
    return filled_arr

def remove_blobs(image_in, blobs, footprint=morphology.ball(3), iterations=1, fill_holes=False):
    # Calculate eroded mask over non-blobs
    blobs_mask = mask_blobs(blobs, image_in.shape)
    nonbgd_mask = ndimage.binary_dilation(blobs_mask, footprint, iterations=iterations)
    
    # Return non-masked values as NaN
    image_out = img_as_float(image_in, force_copy=True)
    image_out[nonbgd_mask] = np.nan
    
    if fill_holes:
        # Interpolate to fill image_out missing NaN holes using distance transform
        if np.any(np.isnan(bgd_mean)):
            image_out = fill_blank_regions(image_out, cval=np.nan)

        # Interpolate to fill any missing NaN edges using nearest neighbor
        if np.any(np.isnan(image_out)):
            image_out = interp_missing(image_out, method='nearest')
    
    return image_out

def get_bgd_images(self, footprint=morphology.ball(3), dilations=1, sigma=1.0):
    image_type_in = 'deskewed'
    image_type_out = 'gaussian'
    data_type_in = 'blobs'
    channels = self.get_channels(image_type_in)
    frames = self.get_frames(image_type_in)
    dset_in = self._movies[image_type_in]
    darr_in = da.stack(dset_in.data_vars.values())
    image_shape = self.metadata['image']['shape']
    
    tasks = [list() for _ in channels]
    for channel_index, channel in enumerate(channels):
        dfs_grouped = self._data[data_type_in][channel].groupby('frame')
        for frame_index, frame in enumerate(frames):
            df_in = dfs_grouped.get_group(frame)
            image = darr_in[channel_index, frame_index]
            blobs = df_in[['z_px', 'y_px', 'x_px', 'sigma_z', 'sigma_y', 'sigma_x']].values

            # Calculate mask to remove dilated blobs from image and fill missing NaN values
            task = dask.delayed(remove_blobs)(image, blobs, footprint, dilations)
            tasks[channel_index].append(task)
    
    # Calculate parallelized result
    futures = dask.persist(*tasks)
    results = dask.compute(*futures)
    
    # Sum the non-blob intensities over the entire movie
    bgd_summed = {channel : np.zeros(image_shape, dtype=float) for channel in channels}
    num_pixels = {channel : np.zeros(image_shape, dtype=int) for channel in channels}
    for channel_index, channel in enumerate(channels):
        for frame_index, frame in enumerate(frames):
            bgd_image = results[channel_index][frame_index]
            bgd_mask = ~np.isnan(bgd_image)
            bgd_summed[channel][bgd_mask] += bgd_image[bgd_mask]
            num_pixels[channel][bgd_mask] += 1
    
    bgd_images = dict.fromkeys(channels)
    for channel in channels:
        # Calculate the mean background over the full (valid) movie
        valid_mask = (num_pixels[channel] > 0)
        bgd_mean = np.nan * np.ones_like(bgd_summed[channel])
        bgd_mean[valid_mask] = bgd_summed[channel][valid_mask] / num_pixels[channel][valid_mask]
        
        # Interpolate to fill any missing NaN holes using distance transform
        if np.any(np.isnan(bgd_mean)):
            bgd_mean = fill_blank_regions(bgd_mean, cval=np.nan)
        
        # Interpolate to fill any missing NaN edges using nearest neighbor
        if np.any(np.isnan(bgd_mean)):
            bgd_mean = interp_missing(bgd_mean, method='nearest')
        
        # Filter the resulting background image
        bgd_images[channel] = ndimage.gaussian_filter(bgd_mean, sigma=sigma)
    
    return bgd_images

def get_cell_mask(bgd_image, classes=3, openings=1, erosions=0):
    # Get multi-Otsu thresholds
    threshold = filters.threshold_multiotsu(bgd_image, classes=classes)[0]
    
    # Segment based on lowest threshold
    cell_mask = bgd_image > threshold
    
    if openings > 0:
        # Open the image for the requested number of iterations
        cell_mask = ndimage.binary_opening(cell_mask, iterations=openings)
    
    if erosions > 0:
        # Erode the image for the requested number of iterations
        cell_mask = ndimage.binary_erosion(cell_mask, iterations=erosions)
    
    return cell_mask

def get_bgd_mask(objs_mask, cell_mask, footprint=None, iterations=1):
    dil_objs_mask = ndimage.binary_dilation(objs_mask, footprint, iterations=iterations)
    bgd_mask = cell_mask & ~dil_objs_mask
    return bgd_mask


def extract_intensities(dset_in, dfs_in, param_dict=dict(), **kwargs):
    # Set parameters and read metadata
    channels = list(dset_in.data_vars)
    
    # Set image metadata
    image_shape = dset_in[channels[0]].shape
    image_dtype = dset_in[channels[0]].dtype
    image_ndim  = dset_in[channels[0]].ndim
    
    # Get input masks
    mask_in = param_dict['images']['skew_mask']
    cell_masks = param_dict['images']['cell_masks']
    bgd_images = param_dict['images']['bgd_images']
    
    # Create Gaussian-filtered images
    lowpass_filter = lambda image : ndimage.gaussian_filter(image, **param_dict['filter_image'])
    images_sig = {channel : lowpass_filter(dset_in[channel].data) for channel in channels}
    
    # Calculate labeled and intensity images for each channel
    results_out = dict.fromkeys(channels)
    for channel1 in channels:
        # Load filtered images and metadata
        image_sig1 = images_sig[channel1]
        bgd_image = bgd_images[channel1]
        
        # Load candidate blobs
        blobs_in = dfs_in[channel1][['z_px', 'y_px', 'x_px', 'sigma_z', 'sigma_y', 'sigma_x']].values
        
        # Filter blobs outside original image
        indexes_keep = filter_blobs(blobs_in, mask_in, **param_dict['filter_blobs'])
        unsorted_blobs = blobs_in[indexes_keep]
        
        # Get unsorted blob coordinates
        unsorted_coords, _ = split_blobs(unsorted_blobs)
        
        # Create array of labeled points
        coms_mask = mask_coords(unsorted_coords, shape=image_shape)
        coms_labeled = label_coords(unsorted_coords, shape=image_shape)
        
        # Return unsorted labels for all blobs (ordered as for current blobs)
        unsorted_labels = coms_labeled[tuple(zip(*unsorted_coords))]
        sorting_indexes = np.argsort(unsorted_labels)

        # Get sorted blob coordinates, sigmas, and radii
        blobs = unsorted_blobs[sorting_indexes]
        coords, sigmas = split_blobs(blobs)
        radii = get_radii(sigmas)

        # Create mask of blobs
        blobs_mask = mask_blobs(blobs, image_shape)

        # Label image using watershed segmentation
        distances = ndimage.distance_transform_edt(blobs_mask)
        blobs_labeled = segmentation.watershed(-distances, coms_labeled, mask=blobs_mask)
        
        # Return final sorted labels for all blobs
        blob_labels = np.array(get_labels(blobs_labeled))
        
        # Get mask of background
        bgd_mask = get_bgd_mask(blobs_mask, cell_masks[channel1], **param_dict['mask_background'])
        
        # Integrate over raw intensities
        raw_sig_ints = labeled_comprehension(image_sig1, blobs_labeled, func=np.sum, labels=blob_labels)

        # Calculate local background intensities
        image_bgd1 = bgd_image * np.mean(image_sig1[bgd_mask]) / np.mean(bgd_image[bgd_mask])
        loc_bgd_ints = labeled_comprehension(image_bgd1, blobs_labeled, func=np.sum, labels=blob_labels)
        
        # Calculate background-subtracted intensity
        bgd_sub_ints = raw_sig_ints - loc_bgd_ints

        # Calculate normalized background-subtracted
        nrm_sig_ints = bgd_sub_ints / loc_bgd_ints
        
        # Filter blobs based on background-subtracted values
        labeled_out = np.zeros(shape=image_shape, dtype=np.uint16)
        indexes_out = list()
        for index, label in enumerate(blob_labels):
            # Get mask for object being tested
            blob_mask = blobs_labeled == label

            # Reject dimmest objects
            if nrm_sig_ints[index] > param_dict['filter_signal'][channel1]['lower']:
                labeled_out[blob_mask] = label
                indexes_out.append(index)
        indexes_out = np.array(indexes_out)
        
        # Save only selected intensities
        raw_sig_ints1 = raw_sig_ints[indexes_out]
        bgd_sub_ints1 = bgd_sub_ints[indexes_out]
        nrm_sig_ints1 = nrm_sig_ints[indexes_out]
        
        # Calculate labels of objects in final result
        labels_out = get_labels(labeled_out)
        coords_out = coords[indexes_out]
        
        # Get coordinates of objects to keep in final result
        z_px, y_px, x_px = np.transpose(coords_out)
        
        # Convert values from pixels to microns
        z_um = z_px * param_dict['px2um']['z']
        y_um = y_px * param_dict['px2um']['y']
        x_um = x_px * param_dict['px2um']['x']
        
        # Return values to retain from given channel
        result_out1 = {'z_px' : z_px, 'y_px' : y_px, 'x_px' : x_px,
                       'z_um' : z_um, 'y_um' : y_um, 'x_um' : x_um,
                       'labels' : labels_out, 
                       'signal_raw_' + channel1 : raw_sig_ints1,
                       'signal_sub_' + channel1 : bgd_sub_ints1,
                       'signal_nrm_' + channel1 : nrm_sig_ints1,
                      }
        
        for channel2 in channels:
            if channel2 == channel1: continue
            
            # Integrate over raw intensities in opposite channel
            image_sig2 = images_sig[channel2]
            raw_sig_ints2 = labeled_comprehension(image_sig2, blobs_labeled, func=np.sum, labels=labels_out)
            
            # Calculate local background intensities in opposite channel
            image_bgd2 = bgd_image * np.mean(image_sig2[bgd_mask]) / np.mean(bgd_image[bgd_mask])
            loc_bgd_ints2 = labeled_comprehension(image_bgd2, blobs_labeled, func=np.sum, labels=labels_out)
            
            # Calculate background-subtracted intensity in opposite channel
            bgd_sub_ints2 = raw_sig_ints2 - loc_bgd_ints2
            
            # Calculate normalized background-subtracted in opposite channel
            nrm_sig_ints2 = bgd_sub_ints2 / loc_bgd_ints2
            
            # Include intensities for opposite channel
            result_out1['signal_raw_' + channel2] = raw_sig_ints2
            result_out1['signal_sub_' + channel2] = bgd_sub_ints2
            result_out1['signal_nrm_' + channel2] = nrm_sig_ints2
        
        # Convert final results to DataFrame
        results_out[channel1] = pd.DataFrame(result_out1)
        
        # Save labeled TIFF image directory to file
        frame = int(dfs_in[channel1]['frame'].mode())
        tif_path = os.path.join(param_dict['labeled_dir'], channel1, f"{channel1}_T{frame:05d}.tif")
        imsave(tif_path, labeled_out.astype(np.uint16))
    
    return results_out


def extract_movie_templates(self, template_radii, ascending=False, num_frames_steps=5, num_templates_per_frame=5):
    frames = self.get_frames()
    channels = self.get_channels()
    image_shape = self.metadata['image']['shape']
    
    # Merge all data for each channel
    ndim = 3
    rounding = 5
    intensity = 'raw'
    px2um = {'z' : 0.210, 'y' : 0.104, 'x' : 0.104}
    axis_names = px2um.keys()
    dfs_dict = self._data
    df_types = dfs_dict.keys()
    channels = dfs_dict[list(df_types)[0]].keys()
    
    dfs_merged = dict()
    for channel in channels:
        # Merge on xyzt data for each particle (in px)
        merge_columns = ['frame', 'z_px', 'y_px', 'x_px']

        # Assume that blobs and intensities are available
        df1 = dfs_dict['blobs'][channel].round(rounding)
        df2 = dfs_dict['intensities'][channel].round(rounding)

        # Merge blobs and intensities data
        df = pd.merge(df1, df2, on=merge_columns)
        
        # Drop duplicates and sort data
        df = df.drop_duplicates(subset=['z_px', 'y_px', 'x_px', 'frame'])
        
        for df_type in df_types:
            try:
                df3 = dfs_dict[df_type][channel].round(rounding)
                if df_type in ('blobs', 'intensities'):
                    # Skip previously merged data
                    continue
                elif df_type in ('tracked', ):
                    # Rename tracked columns
                    rename_columns = {'x' : 'x_um', 'y' : 'y_um', 'z' : 'z_um'}
                    df3.rename(columns=rename_columns, inplace=True)

                    # Rename signal columns
                    old = list(df.filter(like='signal_').columns)
                    new = [f"signal_{intensity}_" + ''.join(o.split('signal_')) for o in old]
                    df3.rename(columns=dict(zip(old, new)), inplace=True)
                    
                    # Merge tracked and intensities/blobs data on xyzt data (in um)
                    merge_columns = ['frame', 'z_um', 'y_um', 'x_um']
                    df = pd.merge(df, df3, on=merge_columns)

                    # Drop duplicates and sort data
                    df = df.drop_duplicates(subset=['track', 'frame'])
                    df = df.sort_values(by=['track', 'frame'])
                else:
                    # Merge based on frame and label
                    df = pd.merge(df, df3, on=['frame', 'labels'])

                    # Sort data
                    df = df.sort_values(by=['frame', 'labels'])
            except Exception as e:
                pass
        
        # Clean up the merged DataFrame
        df = df.dropna()

        # Make a new column for the average radius (in pixels)
        sigma_px = df.filter(regex='sigma').mean(axis=1)
        df['r_px'] = sigma_px * np.sqrt(ndim)

        # Make a new column for the average radius (in microns), assuming correct lateral dimensions
        sigmas_um = df[['sigma_y', 'sigma_x']] * [px2um['y'], px2um['x']]
        df['r_um'] = (sigmas_um * np.sqrt(ndim)).mean(axis=1)

        # Save the merged DataFrame
        dfs_merged[channel] = df
    
    # Write function to calculate source image
    get_image = lambda channel, frame: self._movies['deskewed'][channel].sel(t=frame).data.compute()

    # Get all available frames
    frames = np.sort(list(frames))
    num_frames = len(frames)

    # Set frames to use for templates
    step = num_frames // num_frames_steps
    frames_step = frames[::step]
    
    templates = dict()
    for channel in channels:
        # Get template size for each channel
        radius_by_axis = template_radii[channel]
        template_shape = tuple([2 * r + 1 for r in radius_by_axis])

        # Get data for current channel
        df = dfs_merged[channel]

        # Group data by frame
        df_groups = df.groupby('frame')

        templates_channel = list()
        for frame in frames_step:
            # Get data for current frame
            df_frame = df_groups.get_group(frame)

            # Get current image
            image = get_image(channel, frame)

            # Sort values descending by normalized intensity
            signal_column = 'signal_nrm_' + channel
            df_sorted = df_frame.sort_values(ascending=ascending, by=signal_column)

            # Read out coordinate values (in px)
            coords = df_sorted[['z_px', 'y_px', 'x_px']].values

            # Save templates for current frame
            templates_frame = list()
            for coord in coords:
                slices = find_roi(coord, radius_by_axis, image_shape)
                template = image[slices]
                if template.shape == template_shape:
                    templates_frame.append(template)
                    if len(templates_frame) >= num_templates_per_frame:
                        break

            templates_channel += templates_frame

        # Save all templates and shapes
        templates[channel] = templates_channel
    
    return templates

def profile_line(image, coord1, coord2, spacing=1, order=0, endpoint=True):
    d = spatial.distance.euclidean(coord1, coord2)
    n = int(np.ceil(d / spacing))
    coords = [np.linspace(c1, c2, n, endpoint=endpoint) for c1, c2 in zip(coord1, coord2)]
    profile = ndimage.map_coordinates(image, coords, order=order)
    return profile

def filter_labeled(labeled_in, condition, labels=None):
    if labels is None: labels = get_labels(labeled_in)
    labeled_out = np.zeros_like(labeled_in)
    for index, label in enumerate(labels):
        if condition[index]:
            labeled_out[labeled_in == label] = label
    return labeled_out

# Write function to calculate background-subtracted image
def get_image_bgd_sub(self, channel, frame, normed=True, bgd_images=None, cell_masks=None):
    raw_images = self._movies['deskewed']
    if bgd_images is None: bgd_images = load_tif(movie, 'Background')
    if cell_masks is None: cell_masks = load_tif(movie, 'CellMask')
    
    raw_image = raw_images[channel].sel(t=frame).data.compute()
    bgd_image = bgd_images[channel]
    cell_mask = cell_masks[channel]
    bgd_image = bgd_image * np.mean(raw_image[cell_mask]) / np.mean(bgd_image[cell_mask])
    image_bgd_sub = raw_image - bgd_image
    
    if normed:
        return image_bgd_sub / bgd_image
    else:
        return image_bgd_sub

def cluster_blobs(self, template_radii):
    frames = self.get_frames()
    channels = self.get_channels()

    # Get background image and cell mask
    bgd_images = load_tif(self, 'Background')
    for channel, bgd_image in bgd_images.items():
        bgd_image[bgd_image == 0] = np.nanmean(bgd_image[bgd_image > 0])
        bgd_images[channel] = bgd_image
    cell_masks = load_tif(self, 'CellMask')
    
    # Set template radii based on expected particle sizes
    template_shapes = {c : tuple([2 * r + 1 for r in rs]) for c, rs in template_radii.items()}

    # Get signal and background templates
    templates_sig = extract_movie_templates(self, template_radii, ascending=False, num_frames_steps=5, num_templates_per_frame=2)
    templates_bgd = extract_movie_templates(self, template_radii, ascending=True, num_frames_steps=5, num_templates_per_frame=2)
    
    # Define function to match templates
    @dask.delayed
    def func(image, template, labeled, labels, **kwargs):
        match = feature.match_template(image, template, **kwargs)
        hits = labeled_comprehension(match, labeled, np.max, labels=labels)
        return hits
    
    filters = dict()
    for channel in channels:
        # Use both signal and background templates
        templates = templates_sig[channel] + templates_bgd[channel]
        num_templates = len(templates)

        tasks = list()
        ids = list()
        for frame in frames:
            image = get_image_bgd_sub(self, channel, frame, bgd_images=bgd_images, cell_masks=cell_masks)
            blobs_labeled = self._movies['labeled'][channel].sel(t=frame).data.compute()
            blob_labels = get_labels(blobs_labeled)

            # Find match for each template
            args = blobs_labeled, blob_labels
            kwargs = dict(pad_input=True, mode='edge')
            tasks += [func(image, template, *args, **kwargs) for template in templates]
            ids += [{'frame' : frame, 'labels' : label} for label in blob_labels]
        
        # Compute results for all frames
        with ProgressBar():
            results = dask.compute(*tasks)
        
        # Reshape data as expected
        data = list()
        for index, frame in enumerate(frames):
            start = index * num_templates
            stop = start + num_templates
            data.append(np.vstack(results[start:stop]))
        data = np.transpose(np.hstack(data))
        
        # Calculate k-means clusters
        data = cluster.vq.whiten(data)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        
        # Convert to DataFrame with matching labels
        clustered = pd.DataFrame(ids)
        clustered['cluster'] = kmeans.labels_
        
        # Check that signal is labeled as 1
        nrm_sig_ints = self._data['intensities'][channel].rename(columns={'signal_nrm_' + channel : 'signal'})
        cluster_means = pd.merge(nrm_sig_ints, clustered, on=['frame', 'labels']).groupby('cluster')['signal'].mean()
        if cluster_means.index[np.argmax(cluster_means)] == 0:
            clustered['cluster'] = 1 - kmeans.labels_
    
        # Set filters for all blobs in current channel
        filters[channel] = clustered
    
    return filters
