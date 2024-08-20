"""
Created on Mon Jun 10 13:30:11 2024

@author: yuval

gif reader for analyzing birdsongs
"""

import imageio
import imageio.v3 as iio
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import entropy


def read_gif_frames(gif_path):
    """
    Read GIF frames from a given file path.
    """
    gif = imageio.get_reader(gif_path)
    frames = [frame for frame in gif]
    return frames


def convert_to_grayscale_and_transpose(frames):
    """
    Convert a frame to grayscale for easier processing.
    """
    # gray_frame = np.mean(frame, axis=1)
    gray_frames = np.mean(frames, axis=3, keepdims=True)
    # gray_frames = np.where(gray_frames < 5, 0, gray_frames)
    gray_frames_tp = [np.transpose(g_frame) for g_frame in gray_frames]
    return gray_frames_tp


def calculate_entropy(segment):
    value, counts = np.unique(segment, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities)


def detect_song_phrases(frames, low_filter=5, min_peak_height=10, prominence_range=(10, 70)):
    """
    Detect song phrases in a frame based on changes in pixel values.
    frames : the data array
    low_filter : average pixel values under this threshold will be 0
    min_peak_height : the minimal peak height to be considered a peak
    prominence_range : prominence range values (min, max)
    """
    # avg_pixel_values = [np.mean(frame, axis=1) for frame in frames][0]
    avg_pixel_values = [np.where(np.mean(frame, axis=1) < low_filter, 0, np.mean(frame, axis=1)) for frame in frames][0]
    frame_std = np.std(frames[0], axis=1)

    # boolea_val = frame_std < avg_pixel_values
    # false_peak_indices = np.where(boolea_val)[0]
    # for j in false_peak_indices:
    #     for k in range(j, min(j+50, len(avg_pixel_values))):
    #         avg_pixel_values[k] = 0


    # avg_pixel_values = 
    peaks, properties = find_peaks(avg_pixel_values, height=min_peak_height, distance=1, prominence=prominence_range, width=0.01)

    return peaks, properties


def count_songs(peaks, properties, syllable_gap = 60, nonsong_frame = 200, max_gap=250, min_phrases=2):
    """
    Count the number of songs based on detected peaks.
    peaks : list of peak indices
    properties : dictionary with different properties of the peaks (such as width, left and right bases, height and prominence)
    syllable_gap : max frames gap between each syllable
    nonsong_frame : max frames in a call/noise. This will help us separate those from syllable transitions.
    max_gap : max frames gap between different songs before cutting the sequence
    min_phrases : minimum phrases to be considered a song
    """
    if len(peaks) == 0:
        return 0

    songs_count = 0
    phrase_count = 1
    current_sequence_count = 1 # count the number of consequent syllables
    current_sequence_length = 0 # count the length of current sequence in frames (400 frames = 1 second)
    last_peak = peaks[0]
    for i, peak in enumerate(peaks):
        # initiate variables
        if i == 0:
            # initiate the sequence length with the distance between the first peak left and right bases
            sequence_start = properties['left_bases'][i]
            sequence_end = properties['right_bases'][i]
            current_sequence_length = sequence_end - sequence_start
            continue
        # conditions for different times between last and current peaks, in frames (400 = 1 second):
        if (peak - last_peak) < syllable_gap:  # we are in the same sequence of peaks (=syllables).
            # Increment the length of the current sequence by changing the right base to the base of the current peak
            sequence_end = properties['right_bases'][i]
            current_sequence_length = sequence_end - sequence_start
            current_sequence_count += 1
            last_peak = peak
            continue
        elif (peak - last_peak) > max_gap:  # might be the end of the previous sequence and the start of a new one.
            if current_sequence_count > 8 and (peak - last_peak) < (max_gap + 50):  # if we were in a song and the distance until the next syllable is still reasonable, continue counting(allow greater distance in between, but not greater than 1 second)
                sequence_end = properties['right_bases'][i]
                current_sequence_length = sequence_end - sequence_start
                current_sequence_count += 1
                last_peak = peak
                continue
            if current_sequence_length > nonsong_frame and phrase_count >= min_phrases:  # end of sequence. If the current sequence is longer than the predefined nonsong frame, it is a song.
                #  and current_sequence_count > 7 and np.std(properties['prominences'][i-current_sequence_count+1:i]) < 10
                # print(f"song detected between frames {sequence_start} and {sequence_end}")
                # peak_segment = peaks[properties['left_bases'][0]:properties['left_bases'][-1]]  # Adjust segment size as needed
                peak_entropy = calculate_entropy(peaks[i-current_sequence_count:i-1])
                # print(f"segment {peaks[i-current_sequence_count]} to {peaks[i-1]} entropy = {peak_entropy}, seq_length = {current_sequence_length}")

                if peak_entropy < 2.5 and current_sequence_length < 400: # detect delta function noise to avoid false positive
                    #reset and continue
                    sequence_start = properties['left_bases'][i]
                    sequence_end = properties['right_bases'][i]
                    current_sequence_length = sequence_end - sequence_start
                    current_sequence_count = 1
                    phrase_count = 1
                    continue 
                else:
                    songs_count += 1
                # phrase_count = 1
            
            # reset sequence counter at the end of the sequence:
            sequence_start = properties['left_bases'][i]
            sequence_end = properties['right_bases'][i]
            current_sequence_length = sequence_end - sequence_start
            current_sequence_count = 1
            phrase_count = 1
        else:  # between 60 and 200 frames - possible phrase transition / noise
            if current_sequence_count > 8 or current_sequence_length > nonsong_frame or (properties['widths'][i] > 50 and current_sequence_count > 2):
                # print("phrase transition, continuing the count")
                phrase_count += 1
                sequence_end = properties['right_bases'][i]
                current_sequence_length = sequence_end - sequence_start
                current_sequence_count += 1
                last_peak = peak
                continue
            else:
                # print("non-song detected, resetting counters.")
                # reset sequence counter:
                sequence_start = properties['left_bases'][i]
                sequence_end = properties['right_bases'][i]
                current_sequence_length = sequence_end - sequence_start
                current_sequence_count = 1
                phrase_count = 1

        last_peak = peak

        # Check the last group
    if current_sequence_length > nonsong_frame and phrase_count >= min_phrases:
        peak_entropy = calculate_entropy(peaks[(i-current_sequence_count+1):i])
        # print(f"segment {peaks[i-current_sequence_count+1]} to {peaks[i-1]} entropy = {peak_entropy}, seq_length = {current_sequence_length}")
        if peak_entropy < 2.5 and current_sequence_length < 400: # detect delta function noise to avoid false positive
            return songs_count
        else:   
            songs_count += 1
    # else:
        # print(f"non-song detected between frames {sequence_start} and {sequence_end}. This will not count.")

    return songs_count


def analyze_birdsong(gif_folder, low_filter=5, min_peak_heght=10, prominence_range=(10,70), syllable_gap=60, nonsong_frame=200, max_gap=250, min_phrases=2):
    """
    Analyze all GIF files in a given folder.
    """
    gif_files = glob.glob(os.path.join(gif_folder, "*.gif"))
    # songs_per_day = []
    total_songs = 0
    for gif_file in gif_files:
        frames = read_gif_frames(gif_file)
        frames[0] = frames[0][0:195] # filter out baseline
        gray_frames = convert_to_grayscale_and_transpose(frames)
        peaks, properties = detect_song_phrases(gray_frames[0], low_filter, min_peak_height, prominence_range)
        songs_count = count_songs(peaks, properties, syllable_gap, nonsong_frame, max_gap, min_phrases)
        # songs_per_day.append(songs_count)
        total_songs += songs_count
        # print(f"File: {gif_file} - Number of Songs: {songs_count}")
    
    return total_songs

def arrange_data(data_dict, longest_rec):
    new_dict = dict()
    for key, values in data_dict.items():
        new_val = values
        while len(new_val) < longest_rec:
            new_val.append(0)
        new_dict[key] = new_val
    return new_dict



if __name__ == "__main__":
    print("hello")
    # path_to_dir = r'/Volumes/Labs/cohen/cohenlab/SongRecSmallExpRoom'  # Replace with your gif file path
    # path_to_dir = r'C:/Users/yuval/OneDrive/Desktop/Weizmann/battle_of_the_islands/gifs'
    # bird_names = sorted([os.path.basename(file) for file in os.listdir(path_to_dir) if file != '.DS_Store'])
    bird_names = ['lrrg', 'lo9rrb', 'lgrb', 'lb23', 'rb17', 'lgrry', 'lyrbr41', 'lbrg7', 'lr9rb16', 'ly9rb25', 'ly10rb3']

    print("bird names: ", bird_names)
    frame_rate = 400  # frames per second

    low_filter=8
    min_peak_height=10
    prominence_range=(12, 70) 
    syllable_gap=60
    nonsong_frame=200
    max_gap=400
    min_phrases=2

    params = [low_filter, min_peak_height, prominence_range, syllable_gap, nonsong_frame, max_gap, min_phrases]

    # songs_dict = dict()
    # longest_rec = 0
    # for i, bird in enumerate(bird_names):
    #     songs_per_day = []
    #     path_to_bird_files = os.path.join(path_to_dir, f"{bird}")
    #     dates = sorted([os.path.basename(file) for file in os.listdir(path_to_bird_files) if file != '.DS_Store'])
    #     if len(dates) > longest_rec:
    #         longest_rec = len(dates)
    #     print("dates : ", dates)
    #     print(f"starting to analyze data for bird {bird}:")
    #     for i, date in enumerate(dates):
    #         path_to_curr_day = os.path.join(path_to_bird_files, rf"{date}/chop_data/gif")
    #         songs_per_day.append(analyze_birdsong(path_to_curr_day, *params))
    #         print(f"\t\tfinished calculating number of songs in {date}")
    #     songs_dict[bird] = songs_per_day
    #     print(f"\tfinished calculating number of songs for bird {bird}. Here is the result: {songs_per_day}")

    # df = pd.DataFrame(arrange_data(songs_dict, longest_rec))
    
    # print(df)
    # base_dir = r'/Users/cohenlab/Desktop/battle_of_the_islands/'
    # export_file = r'/Users/cohenlab/Desktop/battle_of_the_islands/britts_scotts_analysis.csv'
    # df.to_csv(export_file, index=True)



# single file / folder analysis:
    birdname = 'lyrbr41'
    # path_to_files = f"/Users/cohenlab/Desktop/battle_of_the_islands/britts_data/{birdname}/2024-05-21/chop_data/gif"
    path_to_files = r'/Users/cohenlab/Desktop/battle_of_the_islands/gifs'
    gif_files = sorted(glob.glob(os.path.join(path_to_files, "*.gif")))
    # print(gif_files[0])
    # print("gif files: ", gif_files)
    total_songs = 0
    # total_songs = analyze_birdsong(path_to_files)
    for i, filename in enumerate(gif_files):
        print(f"file{i}:", os.path.basename(filename)[-12:-4])
        
        frames = read_gif_frames(gif_files[i])
        frames[0] = frames[0][0:195] # filter out baseline

        # convert frames to grayscale and transpose the matrix
        g_frames = convert_to_grayscale_and_transpose(frames)

        # detect phrases       
        peaks, properties = detect_song_phrases(g_frames[0], low_filter, min_peak_height, prominence_range)
        # print("peaks = ", peaks)
       
        songs_count = count_songs(peaks, properties, syllable_gap, nonsong_frame, max_gap, min_phrases)


        if songs_count > 0:
            total_songs += songs_count

        # print(properties.keys())
        props = pd.DataFrame(properties)
        # print(props[props['widths'] > 50].to_string())
        # props.to_csv(r'/Users/cohenlab/Desktop/battle_of_the_islands/properties.csv')

        num_columns = frames[0].shape[1]  # Get the number of columns from the second dimension
        peaks_among_all = np.zeros(num_columns)
        for i, peak_ind in enumerate(peaks):
            peaks_among_all[peak_ind] = properties['peak_heights'][i]

        avg_pixel_values = [np.where(np.mean(frame, axis=1) < low_filter, 0, np.mean(frame, axis=1)) for frame in g_frames[0]]
        # apv = [np.where(np.mean(frame, axis=1) > np.std(frame, axis=1), 1, avg_pixel_values) for frame in g_frames[0]][0]

        frame_std = np.std(g_frames[0][0], axis=1)
        boolea_val = frame_std < avg_pixel_values[0]
        false_peak_indices = np.where(boolea_val)[0]
        print(false_peak_indices)
        for j in false_peak_indices:
            for k in range(j, min(j+50, len(avg_pixel_values[0]))):
                avg_pixel_values[0][k] = 0
        
        # std_pixel_values = [np.std(frame, axis=1) for frame in g_frames[0]]



        # plt.plot(avg_pixel_values[0], color='blue', label='mean_values')
        # # plt.plot(clean_val[0], color='red', label='clean_values')

        # # plt.scatter(np.arange(num_columns), std_pixel_values, s=0.4)
        # plt.plot(peaks_among_all, color='green', label='peaks')
        # plt.legend()
        # plt.show()

        
        if songs_count >= 1:
        #     # print("peaks = ", peaks)
        #     # print("prominence = ", properties['prominences'])
        #     # print("prominence std = ", np.std(properties['prominences']))
        #     # print("widths = ", properties['widths'])
            print(f"Number of songs detected: {songs_count}")

            # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5), sharey=True)

            # ax1.plot(avg_pixel_values[0], color='blue', label='mean_values')
            # ax2.plot(peaks_among_all, color='green', label='peaks')
            

            plt.plot(avg_pixel_values[0], color='blue', label='mean_values') #original
            # plt.plot(clean_val[0], color='red', label='clean_values')

            # plt.scatter(np.arange(num_columns), std_pixel_values, s=0.4)
            plt.plot(peaks_among_all, color='green', label='peaks') # original

            # ax1.set_ylabel("mean signal intensity")
            # ax2.set_ylabel("peak values")
    
            # ax3.set_xlabel("frame number (400 Hz)")
            plt.legend()
            plt.show()


        #     avg_pixel_values = [np.where(np.mean(frame, axis=1) < low_filter, 0, np.mean(frame, axis=1)) for frame in g_frame[0]]
        #     for frame in g_frame[0]:
        #         print(entropy(frame))
        #     plt.plot(avg_pixel_values[0], color='blue', label='mean_values')

        #     # peak_segment = avg_pixel_values[properties['left_bases'][0]:properties['left_bases'][-1]]  # Adjust segment size as needed
        #     # peak_entropy = calculate_entropy(peak_segment)
        #     # print(f"segment entropy ")
        #     plt.plot(peaks_among_all, color='green', label='peaks')
        #     plt.legend()

        #     plt.show()

    print("songs_per_day:", total_songs)    
    print(total_songs)



'''
future analysis - for delta wave exclusion:
calculate peaks that are caused by delta waves by calculating the std of each frame in addition to the mean.
song peaks tend to have a greater std than the mean, while delta waves tend to have smaller std.

    avg_pixel_values = [np.where(np.mean(frame, axis=1) < low_filter, 0, np.mean(frame, axis=1)) for frame in g_frame[0]]
    std_pixel_values = [np.std(frame, axis=1) for frame in g_frame[0]]
    print(f"len avg = {len(avg_pixel_values[0])}, len std = {len(std_pixel_values[0])}")
    new_peaks = peaks
    for i, peak in enumerate(peaks):
        if avg_pixel_values[0][peak] > std_pixel_values[0][peak]:
            print(f"avg = {avg_pixel_values[0][peak]}, std = {std_pixel_values[0][peak]}")
            new_peaks[i] = 0
    for i, peak_ind in enumerate(new_peaks):
        peaks_among_all[peak_ind] = properties['peak_heights'][i]
    # delta = [np.where(np.std(frame, axis=1) < np.mean(frame, axis=1), 0, np.std(frame, axis=1)) for frame in g_frame[0]]
    print("shape:", avg_pixel_values[0].shape)
    print(std_pixel_values)
    plt.plot(avg_pixel_values[0], color='blue', label='mean_values')
    # plt.plot(avg_clean[0], color='red', label='clean_values')
    # plt.scatter(np.arange(num_columns), std_pixel_values, s=0.1)
    plt.plot(peaks_among_all, color='green', label='peaks')
    plt.legend()
    plt.show()
'''
    
