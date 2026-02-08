import os
import pandas as pd
import shutil
import util
from tqdm import tqdm

default_cities = [
    "trondheim",
    "amsterdam",
    "helsinki",
    "tokyo",
    "toronto",
    "saopaulo",
    "moscow",
    "zurich",
    "paris",
    "budapest",
    "austin",
    "berlin",
    "ottawa",
    "goa",
    "amman",
    "nairobi",
    "manila", 
    "bangkok",
    "boston",
    "london",
    "melbourne",
    "phoenix"
]                  
default_files = ["database","query"]

for city in default_cities:
    print(f"Processing {city}...")
    for file_type in default_files:
        # Change to your own dataset path
        csv_file_path_r = os.path.join("/home/lufeng/VPR/datasets_vg/datasets/mapillary_sls/train_val/", city, file_type, "raw.csv")   
        with open(csv_file_path_r, "r") as file:
            raw_lines = file.readlines()[1:]
        # Change to your own destination path   
        dst_folder = os.path.join('/home/lufeng/VPR/Unified_dataset/Mapillary_sls', city, "images", "train") 
        os.makedirs(dst_folder, exist_ok=True)

        for raw_line in tqdm(raw_lines, desc=f"Copy and rename {city}/{file_type} images"):
            _, pano_id, lon, lat, ca, timestamp, is_panorama = raw_line.split(",")
            if is_panorama == "True\n":
                continue
            timestamp = timestamp.replace("-", "")[:-2]
            dst_image_name = util.get_dst_image_name(lat, lon, pano_id, heading=ca, timestamp=timestamp)
            src_image_path = os.path.join(os.path.dirname(csv_file_path_r), 'images', f'{pano_id}.jpg')
            dst_image_path = os.path.join(dst_folder, dst_image_name)
            shutil.copy(src_image_path, dst_image_path)
    print(f"Finished {city}!!!")