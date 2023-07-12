#!/usr/bin/env python

import trackpy as tp

from data_collections import *
from image_analysis import *


def link_tracks(movie, params):
    # Configure channels from data
    channels = movie.get_channels('processed')
    frames = movie.get_frames('processed')

    # Load all available DataFrames
    dfs = movie.merge_dataframes('blobs', 'intensities', 'filters')
    
    for channel, df in dfs.items():
        if params['use_filters']:
            # Filter out likely non-endosomes
            df = df[df['cluster'] == 1]
        
        # Set all columns needed for tracking
        ids_columns = ['frame', 'labels']
        pos_columns = params['trackpy']['pos_columns']
        if params['sig_channels'] == 'same':
            sig_channels = (channel, )
        elif params['sig_channels'] == 'other':
            sig_channels = tuple(set(channels) - set([channel]))
        elif params['sig_channels'] == 'both':
            sig_channels = tuple(channels)
        sig_columns = ['_'.join(['signal', sig_type, channel]) \
                       for sig_type in params['sig_types'] \
                       for channel in sig_channels]

        # Update track parameters
        track_params = copy.deepcopy(params['trackpy'])
        track_params['pos_columns'] = pos_columns + sig_columns

        # Prepare data, normalizing signal intensity
        data_to_track = df[ids_columns + pos_columns + sig_columns]
        for sig_column in sig_columns:
            sig_range = np.ptp(data_to_track[sig_column])
            normalize = track_params['search_range'] / (params['sig_step'] * sig_range)
            data_to_track[sig_column] *= normalize

        # Use trackpy to identify trajectories
        tracked = tp.link(data_to_track, **track_params)

        # Rename columns for ease of use (consistent with SWIFT)
        tracked.rename(columns={'t':'frame'}, inplace=True)
        if tracked['particle'].min() == 0:
            tracked['particle'] += 1

        # Insert new placeholder columnts
        tracked.insert(1, 'track', tracked['particle'])
        tracked.insert(2, 'segment', 1)
        tracked.insert(3, 'parent', 0)
        tracked.sort_values(by=['track', 'frame'], inplace=True)

        # Keep only selected columns
        tracked = tracked[['frame', 'labels', 'track', 'segment', 'parent']]

        # Merge into original dataframe for convenience
        dfs[channel] = df.merge(tracked, on=['frame', 'labels'])
        
    # Only save specific tracked columns
    keep_columns = ['frame', 'track', 'labels', 'segment', 'parent', 'z_um', 'y_um', 'x_um']
    
    # Save data to file
    movie._data['tracked'] = {channel : df[keep_columns] for channel, df in dfs.items()}
    save_csv(movie, 'tracked')

def get_neighbors(movie, use_filters=True, dist_unit='um', px2um={'z':0.210,'y':0.104,'x':0.104},
                  max_surf_dist=2.0, min_overlap=0.0):
    # Load all available DataFrames
    dfs = movie.merge_dataframes('blobs', 'intensities', 'filters', 'tracked')
    
    # Remove untracked and duplicate objects
    tracked = load_csv(movie, 'tracked')
    channels = list(tracked.keys())
    channel1, channel2 = sorted(channels)
    
    for channel, df in dfs.items():
        if use_filters:
            # Drop background particles
            df = df[df['cluster'] > 0]

        # Drop untracked particles or duplicated tracks
        df = df[df['track'] > 0]
        df = df.drop_duplicates(subset=['track', 'frame'])

        dfs[channel] = df
    
    # Determine units of input parameters
    axis_names = px2um.keys()
    
    # Always convert to microns before calculations to ensure appropriate aspect ratios
    if dist_unit == 'px':
        convert = min(px2um.values())
        max_surf_dist *= convert
        dist_unit = 'um'
    
    # Get names of position and radius columns based on distance unit
    pos_columns = ['_'.join([axis_name, dist_unit]) for axis_name in axis_names]
    rad_column = 'r_' + dist_unit

    # Get all relevant frames from both channels
    frames = sorted(set.intersection(*[set(dfs[channel]['frame'].unique()) for channel in channels]))

    # Group trajectories and find track.ids
    df1, df2 = [dfs[channel] for channel in channels]
    grouped1, grouped2 = [dfs[channel].groupby('frame') for channel in channels]

    # Get set of all tracks
    tracks1, tracks2 = [sorted(dfs[channel]['track'].unique()) for channel in channels]

    # Find all track lengths
    lengths1, lengths2 = [dfs[channel]['track'].value_counts() for channel in channels]

    # Catch collisions based on passing within minimum distance of object in other channel
    results = list()
    for frame in tqdm(frames, total=len(frames)):
        # Get all data for current frame
        data1 = grouped1.get_group(frame)
        data2 = grouped2.get_group(frame)

        # Get coordinates in each channel for current frame
        pos1 = data1[pos_columns].values
        pos2 = data2[pos_columns].values

        # Get radii in each channel (assuming perfect spheres)
        rad1 = data1[rad_column].values
        rad2 = data2[rad_column].values
        
        # Grid radii and sum for each pair of endosomes
        rads2, rads1 = np.meshgrid(rad2, rad1)
        rad_sums = rads1 + rads2
        
        # Compute distances between centers of mass
        cent_dists = spatial.distance.cdist(pos1, pos2)

        # Sum radii of each pair of endosomes
        rad_sums = np.sum(np.meshgrid(rad2, rad1), axis=0)

        # Calculate distances of nearest approach at surface
        surf_dists = cent_dists - rad_sums
        
        # Calculate overlap as fraction of each particle's diameter
        overlaps1 = -surf_dists / (2 * rads1)
        overlaps2 = -surf_dists / (2 * rads2)
        
        # Calculate nearest permissible distance between centroids
        min_cent_dist = min_overlap * rad_sums

        # Find all pairs between minimum and maximum distances
        under_max = surf_dists < max_surf_dist
        above_min = cent_dists > min_cent_dist
        
        # Find all pairs less than minimum distance
        indexes1, indexes2 = np.where(under_max & above_min)

        # Save data for each nearby pair (permitting multiple matches per particle)
        for index1, index2 in zip(indexes1, indexes2):
            # Get identity of tracks
            track1 = data1.iloc[index1]['track'].astype(int)
            track2 = data2.iloc[index2]['track'].astype(int)
            if channels[0] == channels[1]:
                if track1 == track2:
                    continue
            
            # Get distance between each centroid and surfaces
            cent_dist12 = cent_dists[index1, index2]
            surf_dist12 = surf_dists[index1, index2]
            
            if surf_dist12 <= 0:
                # Particles are overlapping
                overlap1 = overlaps1[index1, index2]
                overlap2 = overlaps2[index1, index2]
            else:
                # Particles are not overlapping
                overlap1 = 0.
                overlap2 = 0.
            
            # Save data independently for each channel
            result = {'frame' : frame,
                      'track_' + channel1 : track1,
                      'track_' + channel2 : track2,
                      'dist_cent_um' : cent_dist12,
                      'dist_surf_um' : surf_dist12,
                      'overlap_' + channel1 : overlap1,
                      'overlap_' + channel2 : overlap2,
                     }
            
            # Append each collision
            results.append(result)

    # Convert to DataFrame
    results = pd.DataFrame(results)
    
    # Remove duplicate pairs of tracks
    if channels[0] == channels[1]:
        new_index = (pd.DataFrame(data=[sorted(v) for v in results.filter(like='track').values], index=results.index)
                     .drop_duplicates()
                     .index
                    )
        results = results.iloc[new_index]
    
    # Save interim data to file
    movie._data['neighbors'] = {channels[0] : results}
    save_csv(movie, 'neighbors')

def find_collisions(movie, min_track_length=5, frame_rate=2.5,
                    min_frames_between_peaks=3, min_peak_height=0.2):
    # Get all channels and frames
    channels = channels = movie.get_channels()
    channel1, channel2 = sorted(channels)
    frames = movie.get_frames()

    # Load tracked and filtered data
    movie._data['tracked'] = load_csv(movie, 'tracked')
    movie._data['filters'] = load_csv(movie, 'filters')

    # Get data on pairs of neighbors and group by frame
    neighbors = load_csv(movie, 'Neighbors')[channels[0]]
    neighbors_by_frame = neighbors.groupby('frame')

    # Merge DataFrames for required data types
    tracked = movie.merge_dataframes('blobs', 'intensities', 'tracked', 'filters')

    # Get neighbors for current frame
    full_tracked1 = tracked[channels[0]].sort_values(by=['frame'])
    full_tracked2 = tracked[channels[1]].sort_values(by=['frame'])

    # Group tracked data by track
    tracked_by_track1 = full_tracked1.groupby('track')
    tracked_by_track2 = full_tracked2.groupby('track')

    # Get lengths of all tracks
    lengths1 = tracked_by_track1['track'].count().sort_values(ascending=False)
    lengths2 = tracked_by_track2['track'].count().sort_values(ascending=False)

    # Get IDs of all tracks longer than minimum track length
    long_tracks1 = set(lengths1[lengths1 >= min_track_length].index)
    long_tracks2 = set(lengths2[lengths2 >= min_track_length].index)

    # Get only tracked data longer than minimum track length
    long_tracked1 = full_tracked1[full_tracked1['track'].isin(long_tracks1)]
    long_tracked2 = full_tracked2[full_tracked2['track'].isin(long_tracks2)]

    # Get only interactions between particles with longe enough tracks then group by frame
    long_neighbors = neighbors[(neighbors['track_' + channel1].isin(long_tracks1) & 
                                neighbors['track_' + channel2].isin(long_tracks2))]
    long_neighbors_by_frame = long_neighbors.groupby('frame')

    # Group tracks of sufficient length by track ID
    long_tracked_by_track1 = long_tracked1.groupby('track')
    long_tracked_by_track2 = long_tracked2.groupby('track')
    
    @dask.delayed
    def func(long_neighbors_by_frame, frame):
        # Set names of track columns
        suffixes = ['_' + channel for channel in (channel1, channel2)]
        track_columns = ['track' + suffix for suffix in suffixes]

        # Set parameters for merging tracks
        merge_params = dict(how='outer', on='frame', suffixes=suffixes)

        # Set parameters for finding frames of nearest approach
        peaks_params = dict(height=-min_peak_height, distance=min_frames_between_peaks)
        
        results = list()
        try:
            long_neighbors_in_frame = long_neighbors_by_frame.get_group(frame)
        except KeyError:
            return results
        
        for _, (track1, track2) in long_neighbors_in_frame[track_columns].iterrows():
            # Get tracked data for each particle pair
            tracked1 = tracked_by_track1.get_group(track1)
            tracked2 = tracked_by_track2.get_group(track2)

            # Get data to be merged
            keep_columns = ['frame', 'track', 'z_um', 'y_um', 'x_um', 'r_um']
            merge_dfs = tracked1[keep_columns], tracked2[keep_columns]
            merged = pd.merge(*merge_dfs, **merge_params).sort_values(by='frame')

            # Pad merged data with NaNs for absent frames
            filled = (merged
                      .set_index('frame')
                      .reindex(frames)
                      .reset_index()
                      .sort_values(by='frame'))

            # Get full set of position data
            positions = filled.filter(regex='(x|y|z)_um')

            # Calculate velocities from positions and drop NaN values
            velocities = positions.diff(axis=0).dropna() / frame_rate

            # Drop NaN values from position data
            positions.dropna(inplace=True)

            # Get radius data and drop NaNs
            radii = filled.filter(like='r_um').dropna()

            # Get distances between centers of mass
            cent_diffs = (positions.filter(like=channel1).values - \
                          positions.filter(like=channel2).values)
            cent_dists = np.linalg.norm(cent_diffs, axis=1)

            # Get distances between surfaces by subtracting particles sizes
            rad_sums = radii.sum(axis=1)
            surf_dists = (cent_dists - rad_sums.values)

            # Find all frames of closest approach (putative collisions)
            peaks_indexes, _ = signal.find_peaks(-surf_dists, **peaks_params)

            for peak_index in peaks_indexes:
                # Save current frame and distance of closest approach
                nearest_frame = positions.index[peak_index]
                nearest_surf_dist = surf_dists[peak_index]

                result = {k : v for k, v in zip(track_columns, (track1, track2))}
                result = {'frame' : nearest_frame, **result, 
                          'surf_dist' : nearest_surf_dist}
                results.append(result)
        return results

    # Iterate over all frames and tracks within each frame
    tasks = [func(long_neighbors_by_frame, frame) for frame in frames]
    with ProgressBar():
        results = dask.compute(*tasks)
    
    # Combine all results into DataFrame
    results = pd.DataFrame(list(np.hstack(results)))
    
    # Remove duplicate pairs of tracks
    if channels[0] == channels[1]:
        new_index = (pd.DataFrame(data=[sorted(v) for v in results.filter(like='track').values], index=results.index)
                     .drop_duplicates()
                     .index
                    )
        results = results.iloc[new_index]
    
    # Save interim data to file
    movie._data['collisions'] = {channels[0] : results}
    save_csv(movie, 'collisions')

def find_conversions(movie, min_track_length=5, frame_rate=2.5,
                     min_surf_dist=0.2, min_overlap_frames=3, frame_gap=2):
    # Get all channels and frames
    channels = movie.get_channels()
    channel1, channel2 = sorted(channels)
    frames = movie.get_frames()
    
    # Load tracked and filtered data
    movie._data['tracked'] = load_csv(movie, 'tracked')
    movie._data['filters'] = load_csv(movie, 'filters')
    
    # Set names of track columns
    suffixes = ['_' + channel for channel in (channel1, channel2)]
    track_columns = ['track' + suffix for suffix in suffixes]
    
    # Get data on pairs of neighbors and group by frame
    neighbors = load_csv(movie, 'Neighbors')[channel1]
    
    # Merge DataFrames for required data types
    tracked = movie.merge_dataframes('blobs', 'intensities', 'tracked', 'filters')
    
    # Get neighbors for current frame
    full_tracked1 = tracked[channel1].sort_values(by=['frame'])
    full_tracked2 = tracked[channel1].sort_values(by=['frame'])
    
    # Group tracked data by track
    tracked_by_track1 = full_tracked1.groupby('track')
    tracked_by_track2 = full_tracked2.groupby('track')

    # Get lengths of all tracks
    lengths1 = tracked_by_track1['track'].count().sort_values(ascending=False)
    lengths2 = tracked_by_track2['track'].count().sort_values(ascending=False)

    # Get IDs of all tracks longer than minimum track length
    long_tracks1 = set(lengths1[lengths1 >= min_track_length].index)
    long_tracks2 = set(lengths2[lengths2 >= min_track_length].index)

    # Get only tracked data longer than minimum track length
    long_tracked1 = full_tracked1[full_tracked1['track'].isin(long_tracks1)]
    long_tracked2 = full_tracked2[full_tracked2['track'].isin(long_tracks2)]

    # Get only interactions between particles with long enough tracks then group by frame
    long_neighbors = neighbors[(neighbors['track_' + channel1].isin(long_tracks1) & 
                                neighbors['track_' + channel2].isin(long_tracks2))]
    long_neighbors_by_track_pair = long_neighbors.groupby(track_columns)

    # Group tracks of sufficient length by track ID
    long_tracked_by_track1 = long_tracked1.groupby('track')
    long_tracked_by_track2 = long_tracked2.groupby('track')
    
    # Set parameters for merging tracks
    merge_params = dict(how='outer', on='frame', suffixes=suffixes)
    
    # Iterate over all frames and tracks within each frame
    total = len(long_neighbors.groupby(track_columns)['frame'].count())
    results = list()
    for (track1, track2), pair in tqdm(long_neighbors_by_track_pair, total=total):
        if pair['frame'].max() - pair['frame'].min() >= min_track_length:
            overlap = pair[pair['dist_surf_um'] <= min_surf_dist]
            if len(overlap) < min_overlap_frames:
                continue
            
            # Get tracked data for each particle pair
            tracked1 = long_tracked_by_track1.get_group(track1)
            tracked2 = long_tracked_by_track2.get_group(track2)
            
            # Get data to be merged
            keep_columns = ['frame', 'track', 'z_um', 'y_um', 'x_um', 'r_um']
            merge_dfs = tracked1[keep_columns], tracked2[keep_columns]
            merged = pd.merge(*merge_dfs, **merge_params).sort_values(by='frame')

            # Pad merged data with NaNs for absent frames
            filled = (merged
                      .set_index('frame')
                      .reindex(frames)
                      .reset_index()
                      .sort_values(by='frame'))

            # Get full set of position data
            positions = filled.filter(regex='(x|y|z)_um')

            # Calculate velocities from positions and drop NaN values
            velocities = positions.diff(axis=0).dropna() / frame_rate

            # Drop NaN values from position data
            positions.dropna(inplace=True)

            # Get radius data and drop NaNs
            radii = filled.filter(like='r_um').dropna()

            # Get distances between centers of mass
            cent_diffs = (positions.filter(like=channel1).values - \
                          positions.filter(like=channel2).values)
            cent_dists = np.linalg.norm(cent_diffs, axis=1)

            # Get distances between surfaces by subtracting particles sizes
            rad_sums = radii.sum(axis=1)
            surf_dists = (cent_dists - rad_sums.values)
            
            # Get window of time where tracks overlap (below minimum surface distance)
            overlap_frames = set(positions.index[surf_dists <= min_surf_dist])
            has_overlap = filled['frame'].map(lambda f: f in overlap_frames)

            # Eliminate gaps of fewer than specified frame gap to account for missed data
            overlap_window = ndimage.binary_closing(has_overlap.values, iterations=frame_gap)
            overlap_labeled, _ = ndimage.label(overlap_window)
            overlap_labels = np.array(get_labels(overlap_labeled))
            if len(overlap_labels) < 1: continue

            # Keep only overlaps above minimum frame length
            overlap_counts = labeled_comprehension(overlap_window, overlap_labeled, np.size)
            for overlap_label in overlap_labels[overlap_counts >= min_track_length]:
                overlap_keep = filled['frame'][overlap_labeled == overlap_label].values

                result = {k : v for k, v in zip(track_columns, (track1, track2))}
                result = {**result, 'overlap_start' : overlap_keep.min(), 
                          'overlap_stop' : overlap_keep.max(), 
                          'track_' + channel1 + '_start' : tracked1['frame'].min(),
                          'track_' + channel2 + '_start' : tracked2['frame'].min(),
                          'track_' + channel1 + '_stop' : tracked1['frame'].max(),
                          'track_' + channel2 + '_stop' : tracked2['frame'].max(),
                         }
                result['conversion'] = (len(overlap_keep) >= 5) & \
                                        (result[f'track_{channel2}_stop'] - result['overlap_stop'] <= frame_gap)
                results.append(result)

    # Combine all results into DataFrame
    results = pd.DataFrame(results)
    
    # Save interim data to file
    movie._data['conversions'] = {channel1 : results}
    save_csv(movie, 'conversions')
