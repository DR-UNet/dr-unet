import os
import time
import math
import pathlib
from functools import reduce
from collections import Counter

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_path(file_dir):
    path_list = []
    name_list = []
    for path in pathlib.Path(file_dir).iterdir():
        path_list.append(str(path))
        name_list.append(path.name)
    path_list = sorted(path_list, key=lambda path_: int(pathlib.Path(path_).stem))
    name_list = sorted(name_list, key=lambda path_: int(pathlib.Path(path_).stem))
    return path_list, name_list


def check_file(paths):
    if type(paths) is not list:
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return


def list_file(dir_path):
    paths = []
    for path in pathlib.Path(dir_path).iterdir():
        paths.append(str(path))
    return paths


def crop_image(read_dir, save_dir, o_w, o_h, r_w, r_h, split=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Crop large images into small images
    i = 0
    file_paths_list = get_file_path(read_dir)
    for file_path in file_paths_list:
        for row in range(o_h // r_h):
            for col in range(o_w // r_w):
                img = cv.imread(file_path)
                cropped = img[row * r_h: (row + 1) * r_h, col * r_w: (col + 1) * r_w, :]

                if split:
                    save_path = os.path.join(save_dir, str(pathlib.Path(file_path).stem))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), cropped)
                else:
                    cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(i)), cropped)
                i += 1
    print('Cropped image is complete!')
    return
