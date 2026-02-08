
import torch
import shutil
import logging
from typing import Type, List
from argparse import Namespace
from cosface_loss import MarginCosineProduct

import os
import re
import utm
import math


def move_to_device(optimizer: Type[torch.optim.Optimizer], device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(state: dict, is_best: bool, output_folder: str,
                    ckpt_filename: str = "last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state["model_state_dict"], f"{output_folder}/best_model.pth")


def resume_train(args: Namespace, output_folder: str, model: torch.nn.Module,
                 model_optimizer: Type[torch.optim.Optimizer], classifiers: List[MarginCosineProduct],
                 classifiers_optimizers: List[Type[torch.optim.Optimizer]]):
    """Load model, optimizer, and other training parameters"""
    logging.info(f"Loading checkpoint: {args.resume_train}")
    checkpoint = torch.load(args.resume_train)
    start_epoch_num = checkpoint["epoch_num"]
    
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    
    model = model.to(args.device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    assert args.groups_num == len(classifiers) == len(classifiers_optimizers) == \
        len(checkpoint["classifiers_state_dict"]) == len(checkpoint["optimizers_state_dict"]), \
        (f"{args.groups_num}, {len(classifiers)}, {len(classifiers_optimizers)}, "
         f"{len(checkpoint['classifiers_state_dict'])}, {len(checkpoint['optimizers_state_dict'])}")
    
    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        # Move classifiers to GPU before loading their optimizers
        c = c.to(args.device)
        c.load_state_dict(sd)
    for c, sd in zip(classifiers_optimizers, checkpoint["optimizers_state_dict"]):
        c.load_state_dict(sd)
    for c in classifiers:
        # Move classifiers back to CPU to save some GPU memory
        c = c.cpu()
    
    best_val_recall1 = checkpoint["best_val_recall1"]
    
    # Copy best model to current output_folder
    shutil.copy(args.resume_train.replace("last_checkpoint.pth", "best_model.pth"), output_folder)
    
    return model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num


def get_distance(coords_A, coords_B):
    return math.sqrt((float(coords_B[0])-float(coords_A[0]))**2 + (float(coords_B[1])-float(coords_A[1]))**2)


def is_valid_timestamp(timestamp):
    """Return True if it's a valid timestamp, in format YYYYMMDD_hhmmss,
        with all fields from left to right optional.
    >>> is_valid_timestamp('')
    True
    >>> is_valid_timestamp('201901')
    True
    >>> is_valid_timestamp('20190101_123000')
    True
    """
    return bool(re.match("^(\d{4}(\d{2}(\d{2}(_(\d{2})(\d{2})?(\d{2})?)?)?)?)?$", timestamp))

def format_coord(num, left=2, right=5):
    """Return the formatted number as a string with (left) int digits 
            (including sign '-' for negatives) and (right) float digits.
    >>> format_coord2(1.1, 3, 3)
    '001.100'
    >>> format_coord2(-0.12345, 3, 3)
    '-00.123'
    >>> format_coord2(-0.123, 5, 5)
    '-0000.12300'
    """
    return f'{float(num):0={left+right+1}.{right}f}'

import doctest
doctest.testmod()  # Automatically execute unit-test of format_coord()

def format_location_info(latitude, longitude):
    easting, northing, zone_number, zone_letter = utm.from_latlon(float(latitude), float(longitude))
    easting = format_coord(easting, 7, 2)
    northing = format_coord(northing, 7, 2)
    latitude = format_coord(latitude, 3, 5)
    longitude = format_coord(longitude, 4, 5)
    return easting, northing, zone_number, zone_letter, latitude, longitude

def get_dst_image_name(latitude, longitude, pano_id=None, tile_num=None, heading=None,
                       pitch=None, roll=None, height=None, timestamp=None, note=None, extension=".jpg"):
    easting, northing, zone_number, zone_letter, latitude, longitude = format_location_info(latitude, longitude)
    tile_num  = f"{int(float(tile_num)):02d}" if tile_num  is not None else ""
    heading   = f"{int(float(heading))}"      if heading   is not None else ""
    pitch     = f"{int(float(pitch)):03d}"    if pitch     is not None else ""
    timestamp = f"{timestamp}"                if timestamp is not None else ""
    note      = f"{note}"                     if note      is not None else ""
    assert is_valid_timestamp(timestamp), f"{timestamp} is not in YYYYMMDD_hhmmss format"
    if roll is None: roll = ""
    else: raise NotImplementedError()
    if height is None: height = ""
    else: raise NotImplementedError()
    
    return f"@{easting}@{northing}@{zone_number:02d}@{zone_letter}@{latitude}@{longitude}" + \
           f"@{pano_id}@{tile_num}@{heading}@{pitch}@{roll}@{height}@{timestamp}@{note}@{extension}"


