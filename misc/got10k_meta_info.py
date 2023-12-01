import os
import configparser

"""
[METAINFO]
url: https://youtu.be/P-fKM-2DHoM
begin: 00:00:01
end: 00:00:10
anno_fps: 10Hz
object_class: ctenophore
motion_class: swimming
major_class: invertebrate
root_class: animal
motion_adverb: slowly
resolution: (1920, 1080)
"""

# Specify the path to your dataset
dataset_path = r'C:\Users\arthu\OneDrive\Documents\Classwork\EECS542_Adv_computer_vision\got10k\val'

# Create a set to store unique resolutions
unique_resolutions = set()
min_duration = 100000
max_duration = 0

# Loop through each directory in the train folder
for video_folder in os.listdir(dataset_path):
    video_folder_path = os.path.join(dataset_path, video_folder)

    # Check if it's a directory
    if os.path.isdir(video_folder_path):
        meta_info_path = os.path.join(video_folder_path, 'meta_info.ini')

        # Check if meta_info.ini exists in the directory
        if os.path.exists(meta_info_path):
            # Parse the meta_info.ini file
            config = configparser.ConfigParser()
            config.read(meta_info_path)

            # Extract the resolution and add it to the set
            resolution_str = config.get('METAINFO', 'resolution', fallback=None)
            start_time_str = config.get('METAINFO', 'begin', fallback=None)
            end_time_str = config.get('METAINFO', 'end', fallback=None)



            if resolution_str:
                resolution_tuple = eval(resolution_str)
                unique_resolutions.add(resolution_tuple)

            if start_time_str and end_time_str:
                # convert HH:MM:SS to seconds
                start_time = [int(t) for t in start_time_str.split(':')]
                end_time = [int(t) for t in end_time_str.split(':')]
                start_secs = start_time[0] * 3600 + start_time[1] * 60 + start_time[2]
                end_secs = end_time[0] * 3600 + end_time[1] * 60 + end_time[2]
                # print(f"start_time: {start_time_str}, end_time: {end_time_str}, type: {type(start_time_str)}")
                # print(f"start_secs: {start_secs}, end_secs: {end_secs}, type: {type(start_secs)}")
                duration = end_secs - start_secs
                min_duration = min(min_duration, duration)
                max_duration = max(max_duration, duration)
        else:
            print(f"meta_info.ini not found in {video_folder_path}")

# Print the unique resolutions
print("Unique Resolutions:")
aspect_ratios = set()
for resolution in unique_resolutions:
    aspect_ratio = resolution[0] / resolution[1]
    aspect_ratios.add(aspect_ratio)
    print(f"{resolution}, {aspect_ratio:0.4f}")


# Print the min, max and average resolutions
min_resolution = min(unique_resolutions)
max_resolution = max(unique_resolutions)

print(f"Min Resolution: {min_resolution}")
print(f"Max Resolution: {max_resolution}")
print(f"Min Duration: {min_duration} seconds")
print(f"Max Duration: {max_duration} seconds")

"""

Unique Resolutions:
(1920, 1080), 1.7778
(3840, 2160), 1.7778
(720, 480), 1.5000
(1440, 1080), 1.3333
(720, 1280), 0.5625
(480, 360), 1.3333
(640, 480), 1.3333
(960, 720), 1.3333
(1920, 1072), 1.7910
(600, 360), 1.6667
(1280, 720), 1.7778
(640, 360), 1.7778
(848, 480), 1.7667
(854, 480), 1.7792
Min Resolution: (480, 360)
Max Resolution: (3840, 2160)
"""

