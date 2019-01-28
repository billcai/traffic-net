import time
import subprocess

while True:
    print("Now retrieving images from LTA")
    subprocess.call(['python3','./download_image.py',
    '--image_folder','./input_image',
    '--csv_filename','./cameras.csv'])
    print("Now running object detection and refreshing logs")
    subprocess.call(['python3','./detect.py',
    '--input_folder','./input_image',
    '--output_folder','./output_image',
    '--record_folder','./past_records',
    '--all_records','./cameras.csv',
    '--model','./ssd_inference_graph.pb'])
