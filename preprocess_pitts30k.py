import os
import pandas as pd
import shutil
import csv
import random
import string
from tqdm import tqdm

random.seed(0)
unique_names = set()

def generate_name(length=10):
    letters = string.ascii_letters
    name = "".join(random.choice(letters) for _ in range(length))
    return name

# Merge database and query images of pitts30k-train into the same folder
# Change the path to your own dataset path (absolute path is recommended)
source_images_path = "/home/lufeng/VPR/datasets_vg/datasets/pitts30k/images/train/database"
source_images_path1 = "/home/lufeng/VPR/datasets_vg/datasets/pitts30k/images/train/queries"
destination_images_path = "/home/lufeng/VPR/Unified_dataset/pitts30k/images/train"
os.makedirs(destination_images_path, exist_ok=True)

images_list = os.listdir(source_images_path)
images_list1 = os.listdir(source_images_path1)

# copy and rename database images
for filename in tqdm(images_list, desc="Copy and rename database images"):
    src_file_path = os.path.join(source_images_path, filename)
    li = filename.split("@")
    while True:
        id = generate_name(length=10)
        if id not in unique_names:
            unique_names.add(id)
            break
    new_file_name = f"@{li[1]}@{li[2]}@{li[3]}@{li[4]}@{li[5]}@{li[6]}@{id}@@{int(li[8])%12*30}@@@@202409@@.jpg"
    dst_file_path = os.path.join(destination_images_path,new_file_name)
    shutil.copy(src_file_path,dst_file_path)

# copy and rename query images
for filename in tqdm(images_list1, desc="Copy and rename query images"):
    src_file_path1 = os.path.join(source_images_path1, filename)
    li1 = filename.split("@")
    while True:
        id = generate_name(length=10)
        if id not in unique_names:
            unique_names.add(id)
            break
    new_file_name = f"@{li1[1]}@{li1[2]}@{li1[3]}@{li1[4]}@{li1[5]}@{li1[6]}@{id}@@{int(li1[8])%12*30}@@@@202409@@.jpg"
    dst_file_path = os.path.join(destination_images_path,new_file_name)
    shutil.copy(src_file_path1,dst_file_path)