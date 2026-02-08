import pandas as pd
import csv
import numpy as np
import os
import shutil
from collections import defaultdict

header = ['place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid']
default_cities = [
    "Trondheim", 
    "Amsterdam",
    "Helsinki",
    "Tokyo",
    "Toronto",
    "Saopaulo",
    "Moscow",
    "Zurich",
    "Paris",
    "Budapest",
    "Austin",
    "Berlin",
    "Ottawa",
    "Goa",
    "Amman",
    "Nairobi",
    "Manila",
    "bangkok",
    "boston",
    "london",
    "melbourne",
    "phoenix",   # the above cities are from MSLS-train
    "Pitts30k",
    "SFXL",
]

CITYNAME = "SFXL"  # You can change to process other cities in default_cities list
print(f"Processing {CITYNAME}...")

# Due to the ultra-large size of SFLX, we only use one group in our experiments
if CITYNAME == "SFXL":
    NUM_GROUPS = 1
# For Pitts30k and other cities from MSLS-train, we use all groups
# Amman and Nairobi only have 17 groups
elif CITYNAME == "Amman" or CITYNAME == "Nairobi":
    NUM_GROUPS = 17
else:
    NUM_GROUPS = 18

for group in range(NUM_GROUPS):
    # Create CSV file with required columns
    if CITYNAME == "SFXL":
        CITY = "{}".format(CITYNAME)
    else:
        CITY = "{}{}".format(CITYNAME, group)
    with open("{}.csv".format(CITY), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  

    df = pd.read_csv("output{}.csv".format(group),header=None)
    df_target = pd.read_csv("{}.csv".format(CITY))
    classes_id = df[0]
    images_path = df[1]

    city_id_list = []
    place_id_list = []
    lat_list = []
    lon_list = []
    year_list = []
    month_list = []
    name_id_list = []
    northdeg_list = []

    for index, element in enumerate(images_path):
        res = element.split("@")
        
        city_id_list.append(CITY)
        northdeg_list.append(10)
        lat_list.append(res[5])
        lon_list.append(res[6])
        name_id_list.append(res[7])
        year_list.append(res[-3][:4])
        month_list.append(res[-3][4:])
        
    place_id_dict = defaultdict(int)
    for element in classes_id:
        if element not in place_id_dict:
            place_id_dict[element] = 1
        else:
            place_id_dict[element] += 1

    place_id = 0
    for key, value in place_id_dict.items():
        for i in range(value):
            place_id_list.append(place_id)
        place_id += 1

    df_target["city_id"] = city_id_list
    df_target["place_id"] = place_id_list
    df_target["year"] = year_list
    df_target["month"] = month_list
    df_target["northdeg"] = northdeg_list
    df_target["lat"] = lat_list
    df_target["lon"] = lon_list
    df_target["panoid"] = name_id_list

    df_target.to_csv("{}.csv".format(CITY), index=False)
    print("Finshed writing csv file!!!")

    # Copy and Rename Images according to the csv file 
    df_target = pd.read_csv("{}.csv".format(CITY))
    city = df_target["city_id"]
    pl_id = df_target["place_id"]
    year = df_target["year"]
    month = df_target["month"]
    northdeg = df_target["northdeg"]
    lat = df_target["lat"]
    lon = df_target["lon"]
    panoid = df_target["panoid"]

    # Change the path to your own dataset path (absolute path is recommended)
    destination_images_path = "/home/lufeng/VPR/datasets_vg/datasets/new{}/".format(CITYNAME) + CITY
    os.makedirs(destination_images_path, exist_ok=True)

    for index ,element in enumerate(images_path):
        new_file_name = str(city[index]) + "_" + str(pl_id[index]).zfill(7) + "_" + str(year[index]) + "_" + str(month[index]).zfill(2) + "_" + str(northdeg[index]).zfill(3) + "_" + str(lat[index]) + "_" + str(lon[index]) + "_" + panoid[index] + ".jpg"
        target_file = os.path.join(destination_images_path ,new_file_name)
        shutil.copy(element, target_file)
        if index % 100 == 0:
            print(f"finsh {index}")  
            
    # Change the path to your own dataset path (absolute path is recommended)
    shutil.move("{}.csv".format(CITY), "/home/lufeng/VPR/datasets_vg/datasets/new{}".format(CITYNAME)) 
    print("Finshed Group{}!".format(group))