import os
import pathlib

import tqdm
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import utils


def return_inputs(inputs):
    """Returns the output value according to the input type, used for image path input"""
    all_image_paths = None
    if type(inputs) is str:
        if os.path.isfile(inputs):
            all_image_paths = [inputs]
        elif os.path.isdir(inputs):
            all_image_paths = utils.list_file(inputs)
    elif type(inputs) is list:
        all_image_paths = inputs
    return all_image_paths


# 1. make dataset
def get_path_name(data_dir, get_id=False, nums=-1):
    name_list = []
    path_list = []
    for path in pathlib.Path(data_dir).iterdir():
        path_list.append(str(path))
        if get_id:
            name_list.append(path.stem[-5:])
        else:
            name_list.append(path.stem)
    if nums != -1:
        name_list = name_list[:nums]
        path_list = path_list[:nums]
    name_list = sorted(name_list, key=lambda path_: int(pathlib.Path(path_).stem))
    path_list = sorted(path_list, key=lambda path_: int(pathlib.Path(path_).stem))
    return name_list, path_list


class TFData:
    def __init__(self, image_shape, image_dir=None, mask_dir=None,
                 out_name=None, out_dir='', zip_file=True, mask_gray=True):
        self.image_shape = image_shape
        self.zip_file = zip_file
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.out_name = out_name
        self.out_dir = os.path.join(out_dir, out_name)
        self.mask_gray = mask_gray

        if len(image_shape) == 3 and image_shape[-1] != 1:
            self.image_gray = False
        else:
            self.image_gray = True
        if self.zip_file:
            self.options = tf.io.TFRecordOptions(compression_type='GZIP')

        if image_dir is not None and mask_dir is not None:
            self.image_name, self.image_list = get_path_name(self.image_dir, False)
            self.mask_name, self.mask_list = get_path_name(self.mask_dir, False)
            self.data_zip = zip(self.image_list, self.mask_list)

    def image_to_byte(self, path, gray_scale):
        image = cv.imread(path)
        if not gray_scale:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            pass
        image = cv.resize(image, tuple(self.image_shape[:2]))

        return image.tobytes()

    def write_tfrecord(self):
        if not os.path.exists(self.out_dir):
            if self.zip_file:
                writer = tf.io.TFRecordWriter(self.out_dir, self.options)
            else:
                writer = tf.io.TFRecordWriter(self.out_dir)

            print(len(self.image_list))
            for image_path, mask_path in tqdm.tqdm(self.data_zip, total=len(self.image_list)):
                image = self.image_to_byte(image_path, self.image_gray)
                mask = self.image_to_byte(mask_path, self.mask_gray)

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                    }
                ))
                writer.write(example.SerializeToString())
            writer.close()
        print('Dataset finished!')

    def _parse_function(self, example_proto):
        features = tf.io.parse_single_example(
            example_proto,
            features={
                'mask': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string)
            }
        )

        image = features['image']
        image = tf.io.decode_raw(image, tf.uint8)
        if self.image_gray:
            image = tf.reshape(image, self.image_shape[:2])
            image = tf.expand_dims(image, -1)
        else:
            image = tf.reshape(image, self.image_shape)

        label = features['mask']
        label = tf.io.decode_raw(label, tf.uint8)
        if self.mask_gray:
            label = tf.reshape(label, self.image_shape[:2])
            label = tf.expand_dims(label, -1)
        else:
            label = tf.reshape(label, self.image_shape)

        return image, label

    def data_iterator(self, batch_size, data_name='', repeat=1, shuffle=True):
        if len(data_name) == 0:
            data_name = self.out_dir
        else:
            data_name = data_name

        if self.zip_file:
            dataset = tf.data.TFRecordDataset(data_name, compression_type='GZIP')
        else:
            dataset = tf.data.TFRecordDataset(data_name)
        dataset = dataset.map(self._parse_function)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=100).repeat(repeat).batch(batch_size, drop_remainder=True)
        else:
            dataset = dataset.repeat(repeat).batch(batch_size, drop_remainder=True)
        return dataset


def data_preprocess(image, mask):
    """Normalize the image and mask data sets between 0-1"""
    image = tf.cast(image, np.float32)
    image = image / 127.5 - 1
    mask = tf.cast(mask, np.float32)
    mask = mask / 255.0
    return image, mask


def make_data(image_shape, image_dir, mask_dir, out_name=None, out_dir=''):
    tf_data = TFData(image_shape=image_shape, out_dir=out_dir, out_name=out_name,
                     image_dir=image_dir, mask_dir=mask_dir)
    tf_data.write_tfrecord()
    return


def get_tfrecord_data(tf_record_path, tf_record_name, data_shape, batch_size=32, repeat=1, shuffle=True):
    tf_data = TFData(image_shape=data_shape, out_dir=tf_record_path, out_name=tf_record_name)
    seg_data = tf_data.data_iterator(batch_size=batch_size, repeat=repeat, shuffle=shuffle)
    seg_data = seg_data.map(data_preprocess)
    return seg_data


def get_test_data(test_data_path, image_shape, image_nums=16):
    """
    :param test_data_path: test image path
    :param image_shape: Need to resize the shape of the test image, a tuple of length 3, [height, width, channel]
    :param image_nums: How many images need to be tested, the default is 16
    :return: normalized image collection
    """
    or_resize_shape = (1440, 1440)
    normalize_test_data = []
    original_test_data = []
    test_image_name = []
    test_data_paths = return_inputs(test_data_path)

    for path in test_data_paths:
        try:
            test_image_name.append(pathlib.Path(path).name)
            original_test_image = cv.imread(str(path))
            original_test_image = cv.resize(original_test_image, or_resize_shape)
            original_shape = original_test_image.shape
            if len(original_shape) == 0:
                print('Unable to read the {} file, please keep the path without Chinese! --First'.format(str(path)))
            else:
                original_test_data.append(original_test_image)
            if image_shape[-1] == 1:
                original_test_image = cv.cvtColor(original_test_image, cv.COLOR_BGR2GRAY)
            image = cv.resize(original_test_image, tuple(image_shape[:2]))
            image = image.astype(np.float32)
            image = image / 127.5 - 1
            normalize_test_data.append(image)
            if image_nums == -1:
                pass
            else:
                if len(normalize_test_data) == image_nums:
                    break
        except Exception as e:
            print('Unable to read the {} file, please keep the path without Chinese! --Second'.format(str(path)))
            print(e)

    normalize_test_array = np.array(normalize_test_data)
    if image_shape[-1] == 1:
        normalize_test_array = np.expand_dims(normalize_test_array, -1)
    original_test_array = np.array(original_test_data)
    if original_test_array.shape == 3:
        original_test_array = np.expand_dims(original_test_array, 0)
        normalize_test_array = np.expand_dims(normalize_test_array, 0)
    return test_image_name, original_test_array, normalize_test_array
