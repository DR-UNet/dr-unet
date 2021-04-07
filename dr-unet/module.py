from collections import Counter
import pathlib
import math
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
import utils
import tqdm


def binary_image_from_dri(input_dir, threshold=128, save_dir=None):
    if os.path.isdir(input_dir):
        paths = utils.list_file(input_dir)
        utils.check_file([save_dir])
    else:
        paths = [input_dir]

    for path in paths:
        path_stem = pathlib.Path(path).stem
        image = cv.imread(path, 0)
        bin_image = binary_image(image, threshold)
        if save_dir is not None:
            cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(path_stem)), bin_image)
    return


def binary_image(image, threshold):
    shape = image.shape
    if len(shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    th, bin_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return bin_image


def reverse_pred_image(normalize_pred_image):
    reverse_image = normalize_pred_image.squeeze() * 255.0
    reverse_image = np.array(reverse_image, dtype=np.uint8)
    return reverse_image


def save_images(pred, index, save_path, image_shape, split=False):
    image_numbers = int(np.sqrt(pred.shape[0]))
    if not split:
        h = image_shape[0]
        w = image_shape[1]
        H = int(image_numbers * image_shape[0])
        W = int(image_numbers * image_shape[1])
        big_image = np.zeros(shape=(H, W, image_shape[-1]), dtype=np.uint8).squeeze()

        for i in range(pow(image_numbers, 2)):
            image = (pred[i, :, :] * 255.0)
            image = np.array(image, dtype=np.uint8)
            image = image.squeeze()
            j = i % image_numbers
            k = i // image_numbers
            if image_shape[-1] == 1 and len(image_shape) == 3:
                big_image[k * h:(k + 1) * h, j * w:(j + 1) * w] = image
            else:
                big_image[k * h:(k + 1) * h, j * w:(j + 1) * w, :] = image
        path = os.path.join(save_path, 'Segment_train_pred_{}.png'.format(index))
        plt.imsave(path, big_image, cmap='gray')
    else:
        for i in range(image_numbers ** 2):
            image = (pred[i, :, :] * 255.0)
            image = np.array(image, dtype=np.uint8)
            image = image.squeeze()
            path = os.path.join(save_path, 'Segment_pred_{}_{}.png'.format(index, i))
            plt.imsave(path, image, cmap='gray')
    return


def get_area(image):
    """Count the area of bleeding area"""
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, bin_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY)
    count_result = Counter(list(bin_image.reshape(-1, )))
    area = count_result.get(255)
    return area


def pixel_to_ml(pixel_area, dpi=96):
    if pixel_area is None:
        pixel_area = 0.0
    return pixel_area / pow(dpi, 2) * pow(25.4, 2) / 100


def draw_contours(image, mask, max_count=8, dpi=96):
    image = np.array(image)
    mask = np.array(mask)
    height, width = image.shape[:2]
    copy_image = image.copy()

    if len(mask.shape) == 3 and mask.shape[-1] != 1:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    else:
        mask = mask
    mask = cv.resize(mask, (height, width))
    th, bin_mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)

    con_list = []
    blood_area = []
    contours, _ = cv.findContours(bin_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for index, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if 200 < area < height * width * 0.94:
            con_list.append(index)
            blood_area.append(pixel_to_ml(area, dpi))

    if len(con_list) > max_count:
        blood_area = [0]
    else:
        for index in con_list:
            copy_image = cv.drawContours(copy_image, contours, index, (0, 0, 255), 5)
    return copy_image, sum(blood_area)


def save_invalid_data(origin_images, drawed_images, pred_mask_images, image_names, save_dir, reshape=True):
    """
     :param image_names: the image file names of the original images, in the form of a list
     :param origin_images: original bleeding images
     :param drawed_images: draw a contour map of the bleeding area on the original image according to the mask
     :param pred_mask_images: predicted mask image
     :param save_dir: save path of all images
     :param reshape: restore all images to the original image size
     """
    origin_save_dir = os.path.join(save_dir, 'origin')
    drawed_save_dir = os.path.join(save_dir, 'drawed')
    mask_save_dir = os.path.join(save_dir, 'pred_mask')
    utils.check_file([origin_save_dir, drawed_save_dir, mask_save_dir])

    for index in range(len(origin_images)):
        origin_image = origin_images[index]
        drawed_image = drawed_images[index]
        mask_image = pred_mask_images[index]
        _, bin_mask_image = cv.threshold(mask_image, 0, 255, cv.THRESH_BINARY)

        if reshape:
            origin_image = cv.resize(origin_image, (256, 256))
            drawed_image = cv.resize(drawed_image, (256, 256))
            save_name = '{}'.format(image_names[index])
            save_mask_path = os.path.join(mask_save_dir, save_name)
            save_origin_path = os.path.join(origin_save_dir, save_name)
            save_drawed_path = os.path.join(drawed_save_dir, save_name)
            cv.imwrite(save_mask_path, bin_mask_image)
            cv.imwrite(save_origin_path, origin_image)
            cv.imwrite(save_drawed_path, drawed_image)
    return


def count_volume(areas, thickness=0.45):
    for area in areas:
        if area == 0:
            areas.remove(area)
    areas_count = len(areas)
    volume = [areas[index] * thickness for index in range(areas_count)]
    return sum(volume)


def calculate_volume(mask_dir, real_shape=(1440, 1440), thickness=0.4, dpi=96):
    all_areas = []
    for path in tqdm.tqdm(pathlib.Path(mask_dir).iterdir()):
        mask_image = cv.imread(str(path))
        mask_image = cv.resize(mask_image, real_shape)
        area = pixel_to_ml(get_area(mask_image), dpi=dpi)
        all_areas.append(area)
    volume = count_volume(all_areas, thickness)
    return volume
