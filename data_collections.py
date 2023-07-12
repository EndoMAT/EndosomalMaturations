#!/usr/bin/env python

import copy
import glob
import itertools
import json
import logging
import os
import pprint
import re
import shelve
import shutil
import sys
import warnings

import configparser
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import napari
import numpy as np
import pandas as pd
from tifffile import imread, imsave
import xarray as xr

import nest_asyncio
nest_asyncio.apply()

pd.options.mode.chained_assignment = None

# Location of experiment database (should rarely be changed)
DB_PATH = 'database'


def open_database(func):
    """ Decorator function to open database.
    """
    def inner(*args, **kwargs):
        with shelve.open(DB_PATH) as shelf:
            return func(shelf, *args, **kwargs)
    return inner


@open_database
def preview_database(shelf, depth=3):
    """ Preview experiments and movies in current database.
    """
    pprint.pprint(list(shelf.items()), depth=depth)

    
@open_database
def is_in_database(shelf, metadata):
    """ Check database to verify whether movie was successfully logged.
    """
    
    # Check that experiment is in database
    if metadata['names']['experiment'] in shelf:
        expt_dict = shelf[metadata['names']['experiment']]
        
        # Check that movie is in database
        if metadata['names']['movie'] in expt_dict:
            movie_dict = metadata['names']['movie']
            
            # Check that movie dict is not empty
            if len(movie_dict) > 0:
                # Return True only if all conditions are met
                return True
    
    return False


@open_database
def load_database(shelf, expt_names=[]):
    """ Load entire database (or experiments, if specified).
    """
    
    if len(expt_names) == 0:
        # Return full database if experiment not specified
        db_dict = dict(shelf)
    else:
        # Return only subset of database for given experiment names
        db_dict = {expt_name : shelf[expt_name] for expt_name in expt_names}
    
    return db_dict


def load_metadata(*args, **kwargs):
    """ Load metadata for all experiments to be analyzed.
    """
    
    # Load dict of all matching experiments
    db_dict = load_database(*args, **kwargs)
    
    # Return metadata as DataFrame
    metadata = list()
    for expt_name, expt_dict in db_dict.items():
           for movie_name, movie_dict in expt_dict.items():
                try:
                    movie = MultiChannelMovie(**movie_dict)
                    metadata.append({k : movie.metadata['names'][k] for k in ['condition', 'experiment', 'movie']})
                    print(f"Added {movie}.", end=' ' * 50 + '\r')
                except Exception as e:
                    print(f"Error: Could not load {[movie_dict['metadata']['names'][k] for k in ('condition', 'experiment', 'movie')]}. {e}")
    return pd.DataFrame(metadata)


def identify_channel(full_path, channel_aliases):
    """ Attempt to identify channel based on input string (typically file name).
    """
    
    # Only search on file name
    string = os.path.basename(full_path)
    
    # Create regex object based on channel_aliases
    pattern_list = []
    for channel, synonyms in channel_aliases.items():
        pattern_list.append(f"(?P<{channel}>{'|'.join(synonyms)})")
    pattern = '|'.join(pattern_list)
    
    # Search for pattern in file name
    match_obj = re.search(pattern, string, re.IGNORECASE)
    
    # Extract channel if pattern matches exactly once
    channel = None
    if match_obj:
        match_dict = match_obj.groupdict()
        channels = [channel for channel, match in match_dict.items() if match is not None]
        if len(channels) == 1:
            channel = channels[0]
    
    # Test for presence of single channel
    if channel is None:
        if 'EEA1' in string and 'APPL1' not in string:
            channel = 'EEA1'
        elif 'APPL1' in string and 'EEA1' not in string:
            channel = 'APPL1'
    
    # If single channel is not found, require user input
    if channel is None:
        channel = input(f"Could not determine channel automatically from {full_path}. Enter channel (required): ")
    
    return channel


def check_pattern(pattern, string, required=True):
    """ Check whether regex pattern is valid match and request user input if not.
    """
    
    # Find all matches for pattern in input string
    matches = re.findall(pattern, string)
    num_matches = len(matches)
    
    # Update pattern if not one unique match
    if num_matches != 1:
        if num_matches == 0:
            message = f"No match for {pattern} found in {string}. Enter new regex pattern: "
        elif num_matches > 1:
            message = f"Multiple matches for {pattern} found in {string}. Enter new regex pattern: "
        
        # Only get user input if required
        pattern = input(message) if required else ''
    
    return pattern


def import_settings(log_dir, patterns={}):
    """ Import all settings from file.
    """
    
    # File pattern to match name of settings file
    if 'file_name' not in patterns:
        patterns['file_name'] = r"settings*(\.log|\.txt)"
    
    # File pattern to match each new section
    if 'section' not in patterns:
        patterns['section'] = r"^[*\s]+([\w\s]+)[*\s]+$"

    # File pattern to match start of INI file
    if 'config_start' not in patterns:
        patterns['config_start'] = r"\.ini File"

    # Find all matching log files in data directory
    log_files = [name for name in os.listdir(log_dir) 
                 if re.search(patterns['file_name'], name, re.IGNORECASE)]

    if len(log_files) > 1:
        # Default behavior is to only parse first settings file found
        print(f"Multiple settings files found: {', '.join(log_files)}. Reading first file only.")
    elif len(log_files) == 0:
        # Return blank dict if no settings found
        print(f"No settings files found in {log_dir}.", end=' ' * 50 + '\r')
        return dict()

    # Read all lines from setting file
    log_path = os.path.join(log_dir, log_files[0])
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Read settings from text file into dict
    settings = dict()
    ini_file_found = False
    for line_index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if re.search(patterns['config_start'], line, re.IGNORECASE):
            ini_file_found = True
            break

        # Attempt to match line bsaed on section pattern
        section_match = re.match(patterns['section'], line)
        if section_match:
            # Set new top level in dict
            section = section_match.group(1).strip()
            settings[section] = dict()
        else:
            # Settings requires format of {key} : {val}
            colon_index = line.find(':')
            if colon_index is None:
                continue

            # Separate out key and value based on first colon
            key = line[:colon_index].strip()
            val = line[colon_index+1:].strip()
            
            # If any delimiters, split around those
            delims = ('\t', )
            for delim in delims:
                if delim in val:
                    val = tuple([v.strip() for v in val.split(delim)])
            settings[section][key] = val

    # Create custom configuration parser
    config = configparser.RawConfigParser(strict=False)
    config.optionxform = lambda option: option
    config.BOOLEAN_STATES = {'TRUE' : True, 'FALSE': False}

    # All remaining text is considered part of INI file
    config.read_string('\n'.join(lines[line_index+1:]))

    # Save config sections into dict
    settings['Configuration'] = config._sections
    
    return settings


def find_movies(top_dir, channel_aliases, sub_dir='Deskewed', image_pattern='*.tif*', 
                frame_pattern=r"(stack|T)(\d+)", time_pattern=r"(\d+)msec(?!Abs)"):
    """ Search for movies in data directory and return metadata.
    """
    
    # Set directory based on data volume and specified subdirectory
    data_dir = os.path.join(top_dir, sub_dir)
    
    # Loop over all subdirectories (different conditions)
    for condition_name in os.listdir(data_dir):
        condition_dir = os.path.join(data_dir, condition_name)
        if not os.path.isdir(condition_dir): continue
        
        # Loop over all subdirectories (different experiments for each condition)
        for experiment_name in os.listdir(condition_dir):
            experiment_dir = os.path.join(condition_dir, experiment_name)
            if not os.path.isdir(experiment_dir): continue
            
            # Loop over all subdirectories (different movies for each experiment)
            for movie_name in os.listdir(experiment_dir):
                cell_dir = os.path.join(experiment_dir, movie_name)
                if not os.path.isdir(cell_dir): continue
                
                # Identify cameras as valid subdirectories with .tif files for each cell
                camera_names = list()
                channel_names = list()
                image_metadata = dict()
                for camera_name in os.listdir(cell_dir):
                    camera_dir = os.path.join(cell_dir, camera_name)
                    if not os.path.isdir(camera_dir): continue
                    
                    # Set file match pattern
                    glob_pattern = os.path.join(camera_dir, image_pattern)
                    try:
                        # Load sample image to confirm directory validity
                        sample_path = next(glob.iglob(glob_pattern))
                        
                        # Match patterns based only on file name
                        sample_file = os.path.basename(sample_path)
                        
                        # Attempt to identify channel information from image path
                        channel_name = identify_channel(sample_path, channel_aliases)
                        
                        if not image_metadata:
                            # Capture image metadata from sample image
                            sample_image = imread(sample_path)
                            image_metadata = {attr : getattr(sample_image, attr) for attr 
                                              in ('dtype', 'nbytes', 'ndim', 'shape')}
                            
                            # Capture file pattern matches from sample image
                            frame_pattern = check_pattern(frame_pattern, sample_file)
                            time_pattern = check_pattern(time_pattern, sample_file, required=False)
                            
                        # Save camera and channel names
                        camera_names.append(camera_name)
                        channel_names.append(channel_name)
                        
                    except StopIteration:
                        continue
                
                # Skip if valid images not found
                if not image_metadata: continue
                
                # Initialize metadata dict
                metadata = dict.fromkeys(('names', 'paths', 'patterns', 'image', 'settings'))
                
                # Save values to metadata dict
                metadata['names'] = {'condition' : condition_name, 
                                     'experiment' : experiment_name, 
                                     'movie' : movie_name, 
                                     'cameras' : tuple(camera_names), 
                                     'channels' : tuple(channel_names)}
                
                # Set raw data directory
                metadata['paths'] = {'deskewed' : cell_dir}
                
                # Save validated search patterns
                metadata['patterns'] = {'image' : image_pattern, 
                                        'frame' : frame_pattern, 
                                        'time' : time_pattern}
                
                # Set the analysis directories
                for analysis in ('processed', 'labeled', 'tracking'):
                    rel_path = os.path.relpath(cell_dir, data_dir)
                    metadata['paths'][analysis] = os.path.join(top_dir, analysis.title(), rel_path)
                
                # Save image metadata
                metadata['image'] = image_metadata
                
                # Save experiment settings
                metadata['settings'] = import_settings(cell_dir)
                
                yield metadata


def log_movies(*args, overwrite=False, **kwargs):
    """ Find all valid movies and log metadata.
    """
    for metadata in find_movies(*args, **kwargs):
        if overwrite or not is_in_database(metadata):
            try:
                MultiChannelMovie(metadata).save()
            except Exception as e:
                print(f"Error at {metadata['names']['experiment']}, {metadata['names']['movie']}: {e}")


def iter_movies(expt_names=[]):
    """ Iterate over all movies in database (or subset, if specified) in serial.
    """
    
    # Load dict of all matching experiments
    db_dict = load_database(expt_names)
    for expt_name, expt_dict in db_dict.items():
        for movie_name, movie_dict in expt_dict.items():
            yield MultiChannelMovie(**movie_dict)


def apply_movies(movies, func, *args, **kwargs):
    """ Apply function to all movies in database (or subset, if specified) in serial.
    """
    
    # Apply function to all matching experiments
    for movie in tqdm(movies, total=len(movies)):
        try:
            if isinstance(func, str):
                if func == 'map_blocks':
                    method = movie.map_blocks
                elif func =='compute':
                    method = movie.compute
                method(*args, **kwargs)
            else:
                func(movie, *args, **kwargs)
        except Exception as e:
            print(f"Error: {e}.")

            
def save_csv(self, data_type, overwrite=True):
    """ Export DataFrame as CSV in existing tracking directory.
    """
    for channel, df in self._data[data_type].items():
        csv_path = os.path.join(self.metadata['paths']['tracking'], 
                                data_type.title() + '_' + channel + '.csv')
        df.to_csv(csv_path, index=False)

        
def load_csv(self, data_type):
    """ Load DataFrame from CSV in existing tracking directory.
    """
    dfs = dict()
    for channel in self.metadata['names']['channels']:
        try:
            csv_path = os.path.join(self.metadata['paths']['tracking'], 
                                    data_type.title() + '_' + channel + '.csv')
            dfs[channel] = pd.read_csv(csv_path)
        except Exception as e:
            pass
    
    return dfs


def save_tif(self, data_type, images, overwrite=True):
    """ Export DataFrame as CSV in existing tracking directory.
    """
    for channel, image in images.items():
        tif_path = os.path.join(self.metadata['paths']['processed'], 
                                data_type.title() + '_' + channel + '.tif')
        imsave(tif_path, image)

        
def load_tif(self, data_type):
    """ Load DataFrame from CSV in existing tracking directory.
    """
    images = dict()
    for channel in self.metadata['names']['channels']:
        tif_path = os.path.join(self.metadata['paths']['processed'], 
                                data_type.title() + '_' + channel + '.tif')
        images[channel] = imread(tif_path)
    
    return images


def view_image(image, mask=None, colormap='magma'):
    """ View single image (with optional mask) in Napari.
    """
    # Create viewer
    viewer = napari.Viewer(ndisplay=3)

    # Add images to viewer
    viewer.add_image(image, colormap='gray', name='image')
    if mask is not None:
        viewer.add_image(mask, colormap=colormap, name='mask', opacity=0.5)


def view_images(*images, colormap='gray'):
    """ View multiple images in Napari.
    """
    # Create viewer
    viewer = napari.Viewer(ndisplay=3)

    # Add images to viewer
    for index, image in enumerate(images):
        viewer.add_image(image, colormap=colormap, name=index)


class MultiChannelMovie:
    """ Class for holding data corresponding to a multi-channel movie (series of image stacks for 1 or more channels).
    """
    
    def __init__(self, metadata, parameters=dict(), **kwargs):
        # Store object metadata and parameters (if provided)
        self.metadata = metadata
        self.parameters = parameters
        
        # Load additional attributes if provided
        [setattr(self, attr, value) for attr, value in kwargs.items()]
        
        # Initialize dict to hold stacks and corresponding frames
        self._movies = dict()
        self._data = dict()
        self._frames = dict()
        
        # Initialize set to hold types of image data already imported
        self._image_types = set()
        self._data_types = set()
        
        # Load raw data and processed images into delayed Dask array
        image_types = self.metadata['paths'].keys()
        [self.lazy_load_movie(image_type) for image_type in image_types]
        
        # Load analyzed data from CSV files
        data_types = 'blobs', 'intensities', 'tracked'
        for data_type in data_types:
            try:
                channels = self.get_channels()
                data = dict.fromkeys(channels)
                for channel in channels:
                    csv_path = os.path.join(self.metadata['paths']['tracking'], data_type.title() + '_' + channel + '.csv')
                    data[channel] = pd.read_csv(csv_path)
                self._data[data_type] = data
                self._data_types.add(data_type)
            except FileNotFoundError:
                pass
        
        # Provide access to all available frame numbers
        self.frames = set.union(*map(set, self._frames.values()))
        self.num_frames = len(self.frames)
    
    def __getattr__(self, attr):
        if attr in self._image_types:
            return self._movies[attr]
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr))
    
    def __len__(self):
        return self.num_frames
    
    def __str__(self):
        return f"{type(self).__name__} of {self.metadata['names']['experiment']} {self.metadata['names']['movie']}"
    
    def _get_full_path(self, image_type, channel_index):
        # Return full path according to data type (cameras for preprocessed, channels for postprocessed)
        if image_type in ('raw_data', 'deskewed'):
            sub_dir = self.metadata['names']['cameras'][channel_index]
        else:
            sub_dir = self.metadata['names']['channels'][channel_index]
        
        return os.path.join(self.metadata['paths'][image_type], sub_dir)
    
    def _find_images(self, image_type, channel_index):
        # Create file search pattern to find all images for specified channel
        glob_pattern = os.path.join(self._get_full_path(image_type, channel_index), self.metadata['patterns']['image'])
        
        # Read all image paths and sort lexigraphically
        image_paths = sorted(glob.glob(glob_pattern))
        
        # Collect frame numbers in order to properly sort images
        frame_map = dict.fromkeys(range(len(image_paths)))
        for image_index, image_path in enumerate(image_paths):
            # Get name of current image
            image_file = os.path.basename(image_path)
            
            # Attempt to parse frame number from image name
            match = re.search(self.metadata['patterns']['frame'], image_file)
            if match and len(match.groups()) == 2:
                frame_num = int(match.group(2))
                frame_map[frame_num] = image_index
            else:
                print(f"Valid frame number not found for {image_path}. Skipping file import...")
                continue
        
        # Store sorted frame numbers only for deskewed data
        frames = sorted(frame_map.keys())
        
        # Return paths to all valid images
        for frame in frames:
            yield frame, image_paths[frame_map[frame]]
    
    def get_channels(self, image_type='deskewed'):
        return list(self._movies[image_type].data_vars)
        
    def get_frames(self, image_type='deskewed'):
        return self._movies[image_type]['t'].values
    
    def save(self):
        # Store metadata and parameters, indexed by experiment and movie
        expt_name = self.metadata['names']['experiment']
        movie_name = self.metadata['names']['movie']
        
        with shelve.open(DB_PATH) as shelf:
            # Load experiment dict or create new entry
            if expt_name not in shelf:
                expt_dict = dict()
            else:
                expt_dict = shelf[expt_name]

            # Save specified attributes only
            attrs = 'metadata', 'parameters'
            movie_dict = dict.fromkeys(attrs)
            for attr in attrs:
                movie_dict[attr] = getattr(self, attr)

            # Save results back to file
            expt_dict[movie_name] = movie_dict
            shelf[expt_name] = expt_dict
    
    @classmethod
    def load(cls, expt_name, movie_name):
        # Create object from stored attributes
        with shelve.open(DB_PATH) as shelf:
            movie_dict = shelf[expt_name][movie_name]
        
        return cls(**movie_dict)
    
    def load_data(self, *data_types):
        # Load DataFrame from CSV in existing tracking directory
        for data_type in data_types:
            self._data[data_type] = dict()
            for channel in self.metadata['names']['channels']:
                try:
                    csv_path = os.path.join(self.metadata['paths']['tracking'], 
                                            data_type.title() + '_' + channel + '.csv')
                    df = pd.read_csv(csv_path)
                    self._data[data_type][channel] = df
                except FileNotFoundError as e:
                    print(e)
    
    def lazy_load_movie(self, image_type, verbose=True):
        if image_type in ('deskewed', 'labeled'):
            # Load image data from file using Dask delayed array
            da_params = {'shape' : self.metadata['image']['shape'], 
                         'dtype' : self.metadata['image']['dtype']}

            # Load stacks for each movie as Dask delayed arrays, indexed by frame
            movie_dicts = list()
            channels = list()
            for channel_index, channel_name in enumerate(self.metadata['names']['channels']):
                movie_dict = dict()
                for frame, image_path in self._find_images(image_type, channel_index):
                    image_data = da.from_delayed(dask.delayed(imread)(image_path), **da_params)
                    movie_dict[frame] = image_data
                movie_dicts.append(movie_dict)
                channels.append(channel_name)

            # Extract sorted list of frames common to all channels
            frames = sorted(set.intersection(*map(set, movie_dicts)))
            
            if frames:
                # Convert dict to Dask array, preserving frame order
                da_arr = da.stack([[movie_dict[frame] for frame in frames] for movie_dict in movie_dicts])
                
                # Convert to Xarray DataArray
                xr_arr = xr.DataArray(data=da_arr, dims=['c', 't', 'z', 'y', 'x'], 
                                      coords={'c' : channels, 't' : frames}, attrs={'image_type' : image_type})
                
                # Convert to Xarray Dataset
                dset = xr_arr.to_dataset(dim='c')
                
                # Store results only if there is a nonzero number of frames
                self._frames[image_type] = frames
                self._movies[image_type] = dset
                self._image_types.add(image_type)
        else:
            try:
                # Load directly from Zarr file
                dset_path = self.metadata['paths'][image_type]
                chunks = dict(self._movies['deskewed'].chunks)
                self._movies[image_type] = xr.open_dataset(dset_path, engine='zarr', chunks=chunks, consolidated=False)
                self._image_types.add(image_type)
            except:
                pass
    
    def map_blocks(self, func, image_type_in, image_type_out, *args, param_dict=dict(), to_disk=True, overwrite=False, **kwargs):
        # Convert data variables from Dataset into Dask arrays to use Dask Array map_blocks function
        channels = self.get_channels(image_type_in)
        frames = self.get_frames(image_type_in)
        dset_in = self._movies[image_type_in]
        
        # Use Dask array to map blocks
        darr_out = list()
        for channel, darr_in in dset_in.data_vars.items():
            # Include dtype and chunks if not provided
            kwargs = {
                **{'dtype' : self.metadata['image']['dtype'], 'chunks' : darr_in.chunks},
                **(param_dict[channel] if channel in param_dict else dict()),
                **kwargs,
            }
            
            # Use Dask array to map blocks
            darr_out.append(da.map_blocks(func, darr_in.data, *args, **kwargs))
        darr_out = da.stack(darr_out)
        
        # Convert to Xarray DataArray
        xr_arr_out = xr.DataArray(data=darr_out, dims=['c', 't', 'z', 'y', 'x'], 
                                  coords={'c' : channels, 't' : frames}, 
                                  attrs={'image_type' : image_type_out})
        
        # Convert to Xarray Dataset
        dset_out = xr_arr_out.to_dataset(dim='c')
        
        # Set the movie attributes
        self._movies[image_type_out] = dset_out
        self._image_types.add(image_type_out)
        
        # Save current parameters
        self.parameters[func.__name__] = dict(kwargs)
        
        # Save immediately to disk if specified, or always if overwriting previous results
        to_disk and self.save_movie(image_type_out, overwrite=overwrite)
        
    def compute(self, func, image_type_in, data_type_in, data_type_out, *args, param_dict=dict(), overwrite=False, **kwargs):
        # Check existence of CSV files before processing
        channels = self.get_channels(image_type_in)
        frames = self.get_frames(image_type_in)
        csv_paths = dict.fromkeys(channels)
        for channel in channels:
            csv_paths[channel] = os.path.join(self.metadata['paths']['tracking'], 
                                              data_type_out.title() + '_' + channel + '.csv')
        if not overwrite and all([os.path.exists(csv_path) for csv_path in csv_paths.values()]):
            print(f"Skipping {data_type_out} of {str(self)}...")
            return
        
        print(f"Calculating {data_type_out} of {str(self)}...")
        
        # Function must be applied to each frame individually
        tasks = []
        for frame in frames:
            # Read data variables for current frame
            data = [self._movies[image_type_in].sel(t=frame)]
            
            if data_type_in:
                dfs_in = {channel : self._data[data_type_in][channel].groupby('frame').get_group(frame) 
                          for channel in channels}
                data.append(dfs_in)
            
            task = dask.delayed(func)(*data, *args, param_dict=param_dict, **kwargs)
            tasks.append(task)

        # Calculate parallelized result
        with ProgressBar():
            futures = dask.persist(*tasks)
        results = dask.compute(*futures)
        
        # Convert to dict of DataFrames
        dict_out = dict.fromkeys(channels)
        for channel in channels:
            dfs = [None for _ in frames]
            for index, frame in enumerate(frames):
                df = results[index][channel]
                df.insert(0, 'frame', frame)
                dfs[frame] = df

            # Concatenate DataFrames together
            dict_out[channel] = pd.concat(dfs)
        
        # Save final results as movie data
        self._data[data_type_out] = dict_out
        
        # Export results to separate CSV for each channel
        for channel, df in dict_out.items():
            csv_path = csv_paths[channel]
            csv_dir = os.path.dirname(csv_path)
            not os.path.exists(csv_dir) and os.makedirs(csv_dir)
            df.to_csv(csv_path, index=False)
        
        # Set the data types attributes
        self._data_types.add(data_type_out)

        # Save current parameters
        self.parameters[func.__name__] = dict(kwargs)
        
        # Save updated parameters
        self.save()
        
    def save_movie(self, image_type, overwrite=False, *args, **kwargs):
        # Save requested image type to Zarr file
        dset_path = self.metadata['paths'][image_type]
        
        # Create directory structuring, optionally overwriting existing files
        if os.path.exists(dset_path):
            if not overwrite and len(os.listdir(dset_path)) > 0:
                print(f"Not overwriting {image_type} data of {str(self)}...")
                return
        else:
            os.makedirs(dset_path)
        
        # Compute and save results to disk
        print(f"Computing and storing {image_type} data of {str(self)}...")
        with ProgressBar():
            self._movies[image_type].to_zarr(dset_path, *args, **kwargs)
        
        # Save updated parameters
        self.save()
    
    @staticmethod
    def set_view_params(movie_types, channels):
        colormaps = {'APPL1' : 'cyan', 'EEA1' : 'magenta', 'Rab5' : 'cyan', 'Rab7' : 'magenta', 'EGF' : 'yellow', 'Lamp1' : 'magenta'}

        params = {movie_type : dict() for movie_type in movie_types}
        for movie_type in movie_types:
            params[movie_type] = [dict() for _ in channels]

            for channel_index, channel in enumerate(channels):
                # Set parameters that apply to all movie types and channnels
                params[movie_type][channel_index]['blending'] = 'additive'
                params[movie_type][channel_index]['name'] = '_'.join([channel, movie_type])

                # Set parameters that apply to all movie types but specific channels
                params[movie_type][channel_index]['colormap'] = colormaps[channel]

        # Set parameters that apply to specific movie types but all channels
        for channel_index in range(len(channels)):
            params['processed'][channel_index]['gamma'] = 1.5
            params['processed'][channel_index]['multiscale'] = False
            params['processed'][channel_index]['opacity'] = 0.9

            params['labeled'][channel_index]['gamma'] = 0.01
            params['labeled'][channel_index]['interpolation'] = 'nearest'
            params['labeled'][channel_index]['opacity'] = 0.2

        return params
    
    def filter_labels(self, frames=[]):
        # Set channel and frame identities
        channels = self.get_channels()
        if len(frames) == 0:
            frames = sorted(self.get_frames())
        else:
            frames = sorted(set(frames) & set(self.get_frames()))
        
        # Get data for unfiltered labels
        unfiltered = self.labeled.sel(t=frames).to_array(dim='c').data.compute()
        
        # Filter to only likely endosomes
        self.load_data('filters')

        @dask.delayed
        def func(arr_in, df):
            # Keep only items marked as likely endosomes
            arr_out = np.zeros_like(arr_in)
            for label in df['labels'][df['cluster'] == 1].values:
                arr_out[arr_in == label] = label
            return arr_out

        # Apply function to each frame and channel independently
        tasks = list()
        for channel_index, (channel, df) in enumerate(self._data['filters'].items()):
            by_frame = df[df['frame'].isin(frames)].sort_values(by='frame').groupby('frame')
            for frame_index, (frame, group) in enumerate(by_frame):
                tasks.append(func(unfiltered[channel_index, frame_index], group))

        with ProgressBar():
            results = dask.compute(*tasks)

        # Assign filtered particles in parallel
        filtered = np.zeros_like(unfiltered)
        channel_iter = range(len(channels))
        frame_iter = range(len(frames))
        task_iter = enumerate(itertools.product(channel_iter, frame_iter))
        for task_index, (channel_index, frame_index) in task_iter:
            filtered[channel_index, frame_index] = results[task_index]

        return filtered
    
    def view_labels(self, use_filters=False, frames=[], params=dict(), debugging=False):
        if len(frames) == 0:
            frames = sorted(self.get_frames())
        else:
            frames = sorted(set(frames) & set(self.get_frames()))
        
        # Load processed and labeled data
        movie_types = 'processed', 'labeled'
        stacks = dict.fromkeys(movie_types)
        stacks['processed'] = self.processed.sel(t=frames).to_array(dim='c')
        if use_filters:
            stacks['labeled'] = self.filter_labels(frames=frames)
        else:
            stacks['labeled'] = self.labeled.sel(t=frames).to_array(dim='c').data.compute()

        # Set channels
        channels = self.get_channels()

        # Set view parameters
        if not params:
            params = MultiChannelMovie.set_view_params(movie_types, channels)

        # Initialize viewer
        viewer = napari.Viewer(ndisplay=self.metadata['image']['ndim'])

        # Add images for all channels and movie types
        view_iter = itertools.product(reversed(movie_types), reversed(range(len(channels))))
        for movie_type, channel_index in view_iter:
            stack = stacks[movie_type][channel_index]
            param = params[movie_type][channel_index]
            viewer.add_image(stack, **param)

        # Scale view to match real dimensions
        for layer in viewer.layers:
            layer.scale = 0.21, 0.104, 0.104

        # Adjust camera settings
        viewer.camera.zoom = 1.5 / 0.104
        viewer.camera.center = 10, 20, 30
        viewer.camera.angles = 15, 45, 150

        napari.run()

        if debugging:
            return viewer

    def merge_dataframes(self, *df_types, rounding=5, intensity='raw'):
        ndim = 3
        axis_names = 'z', 'y', 'x'
        px2um = {'z' : 0.210, 'y' : 0.104, 'x' : 0.104}

        dfs_dict = self._data
        if len(df_types) < 1:
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
                        df3.rename(columns={'x':'x_um', 'y':'y_um', 'z':'z_um'}, inplace=True)
                        
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

        return dfs_merged
