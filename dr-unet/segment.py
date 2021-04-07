import os
import time
import argparse
import pathlib

import tqdm
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# Custom package
import data
import loss
import utils
import module
import performance
from model import dr_unet

config = ConfigProto()
config.gpu_options.allow_growth = True

# 1. Parameter settings
parser = argparse.ArgumentParser(description="Segment Use Args")
parser.add_argument('--model-name', default='DR_UNet', type=str)
parser.add_argument('--dims', default=32, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--lr', default=2e-4, type=float)

# Training data, testing, verification parameter settings
parser.add_argument('--height', default=256, type=int)
parser.add_argument('--width', default=256, type=int)
parser.add_argument('--channel', default=1, type=int)
parser.add_argument('--pred-height', default=4 * 256, type=int)
parser.add_argument('--pred-width', default=4 * 256, type=int)
parser.add_argument('--total-samples', default=5000, type=int)
parser.add_argument('--invalid-samples', default=1000, type=int)
parser.add_argument('--regularize', default=False, type=bool)
parser.add_argument('--record-dir', default=r'', type=str, help='the save dir of tfrecord')
parser.add_argument('--train-record-name', type=str, default=r'train_data', help='the train record save name')
parser.add_argument('--test-image-dir', default=r'', type=str, help='the path of test images dir')
parser.add_argument('--invalid-record-name', type=str, default=r'test_data', help='the invalid record save name')
parser.add_argument('--gt-mask-dir', default=r'', type=str, help='the ground truth dir of validation set')
parser.add_argument('--invalid-volume-dir', default=r'', type=str, help='estimation bleeding volume')
args = parser.parse_args()


class Segmentation:
    def __init__(self, params):
        self.params = params
        self.input_shape = [params.height, params.width, params.channel]
        self.mask_shape = [params.height, params.width, 1]
        self.model_name = params.model_name
        self.crop_height = params.pred_height
        self.crop_width = params.pred_width
        self.regularize = params.regularize

        # Obtain a segmentation model
        self.seg_model = dr_unet.dr_unet(input_shape=self.input_shape, dims=params.dims)
        self.seg_model.summary()

        # Optimization function
        self.optimizer = tf.keras.optimizers.Adam(lr=params.lr)

        # Every epoch, predict invalid-images to test the segmentation performance of the model
        self.save_dir = str(params.model_name).upper()
        self.weight_save_dir = os.path.join(self.save_dir, 'checkpoint')
        self.pred_invalid_save_dir = os.path.join(self.save_dir, 'invalid_pred')
        self.invalid_crop_save_dir = os.path.join(self.save_dir, 'invalid_pred_crop')
        self.pred_test_save_dir = os.path.join(self.save_dir, 'test_pred')
        utils.check_file([
            self.save_dir, self.weight_save_dir, self.pred_invalid_save_dir,
            self.pred_test_save_dir, self.invalid_crop_save_dir]
        )

        # Save model parameters
        train_steps = tf.Variable(0, tf.int32)
        self.save_ckpt = tf.train.Checkpoint(
            train_steps=train_steps, seg_model=self.seg_model, model_optimizer=self.optimizer)
        self.save_manger = tf.train.CheckpointManager(
            self.save_ckpt, directory=self.weight_save_dir, max_to_keep=1)

        # Set the loss function
        self.loss_fun = loss.bce_dice_loss

    def load_model(self):
        if self.save_manger.latest_checkpoint:
            self.save_ckpt.restore(self.save_manger.latest_checkpoint)
            print('Loading model: {}'.format(self.save_manger.latest_checkpoint))
        else:
            print('Retrain the model！')
        return

    @tf.function
    def train_step(self, inputs, target):
        tf.keras.backend.set_learning_phase(True)

        with tf.GradientTape() as tape:
            pred_mask = self.seg_model(inputs)
            loss = self.loss_fun(target, pred_mask)
            if self.regularize:
                loss = tf.reduce_sum(loss) + tf.reduce_sum(self.seg_model.losses)
        gradient = tape.gradient(loss, self.seg_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.seg_model.trainable_variables))
        return tf.reduce_mean(loss)

    @tf.function
    def inference(self, inputs):
        tf.keras.backend.set_learning_phase(True)
        pred = self.seg_model(inputs)
        return pred

    @staticmethod
    def calculate_volume_by_mask(mask_dir, save_dir, model_name, dpi=96, thickness=0.45):
        all_mask_file_paths = utils.list_file(mask_dir)

        pd_record = pd.DataFrame(columns=['file_name', 'Volume'])
        for file_dir in tqdm.tqdm(all_mask_file_paths):
            file_name = pathlib.Path(file_dir).stem

            each_blood_volume = module.calculate_volume(file_dir, thickness=thickness, dpi=dpi)
            pd_record = pd_record.append({'file_name': file_name, 'Volume': each_blood_volume}, ignore_index=True)
            pd_record.to_csv(
                os.path.join(save_dir, '{}_{}.csv'.format(model_name, file_name)), index=True, header=True)
        return

    def predict_blood_volume(self, input_dir, save_dir, calc_nums=-1, dpi=96, thickness=0.45):
        """
         :param input_dir: The directory for testing bleeding volume images,
                           there are multiple folders under the directory, each folder represents a CT image of a patient
         :param save_dir: The predicted segmented image save directory
         :param calc_nums: predict how many images in the folder
         :param dpi: image parameters
         :param thickness: slice thickness
        """
        # Loading weights of model
        self.load_model()
        save_pred_images_dir = os.path.join(save_dir, 'pred_images')
        save_pred_csv_dir = os.path.join(save_dir, 'pred_csv')
        utils.check_file([save_pred_images_dir, save_pred_csv_dir])
        all_file_dirs = utils.list_file(input_dir)

        cost_time_list = []
        total_images = 0
        for file_dir in tqdm.tqdm(all_file_dirs):
            file_name = pathlib.Path(file_dir).stem

            image_names, ori_images, normed_images = data.get_test_data(
                test_data_path=file_dir, image_shape=self.input_shape, image_nums=calc_nums)
            total_images += len(image_names)

            start_time = time.time()
            pred_mask = self.inference(normed_images)
            end_time = time.time()
            print('FPS: {}'.format(pred_mask.shape[0] / (end_time - start_time)),
                  pred_mask.shape, end_time - start_time)

            denorm_pred_mask = module.reverse_pred_image(pred_mask.numpy())  # (image_nums, 256, 256, 1)
            if denorm_pred_mask.ndim == 2 and self.input_shape[-1] == 1:
                denorm_pred_mask = np.expand_dims(denorm_pred_mask, 0)

            drawed_images = []
            blood_areas = []
            pd_record = pd.DataFrame(columns=['image_name', 'Square Centimeter', 'Volume'])
            for index in range(denorm_pred_mask.shape[0]):
                drawed_image, blood_area = module.draw_contours(ori_images[index], denorm_pred_mask[index], dpi=dpi)
                drawed_images.append(drawed_image)
                blood_areas.append(blood_area)
                pd_record = pd_record.append({'image_name': image_names[index], 'Square Centimeter': blood_area},
                                             ignore_index=True)

            one_pred_save_dir = os.path.join(save_pred_images_dir, file_name)
            module.save_invalid_data(ori_images, drawed_images, denorm_pred_mask,
                                     image_names, reshape=True, save_dir=one_pred_save_dir)

            # Calculate the amount of bleeding based on the area of each layer of hematoma
            blood_volume = module.count_volume(blood_areas, thickness=thickness)
            pd_record = pd_record.append({'Volume': blood_volume}, ignore_index=True)
            pd_record.to_csv(os.path.join(save_pred_csv_dir, '{}_{}.csv'.format(self.model_name, file_name)),
                             index=True, header=True)
            cost_time_list.append(end_time - start_time)
            print('FileName: {} time: {}'.format(file_name, end_time - start_time))
        print('total_time: {:.2f}, mean_time: {:.2f}, total_images: {}'.format(
            np.sum(cost_time_list), np.mean(cost_time_list), total_images))
        return

    def predict_and_save(self, input_dir, save_dir, calc_nums=-1, batch_size=16):
        """ predict bleeding image and save
         :param input_dir: There are several images waiting to be tested under the input_dir folder
         :param save_dir: The file directory where the segmented image predicted by the model is saved
         :param calc_nums: How many images are taken from the directory to participate in the calculation
         :param batch_size: how many images to test each time
         :return:
        """
        mask_save_dir = os.path.join(save_dir, 'pred_mask')
        drawed_save_dir = os.path.join(save_dir, 'drawed_image')
        utils.check_file([mask_save_dir, drawed_save_dir])
        self.load_model()

        test_image_list = utils.list_file(input_dir)
        for index in range(len(test_image_list) // 128 + 1):
            input_test_list = test_image_list[index * 128:(index + 1) * 128]

            image_names, ori_images, normed_images = data.get_test_data(
                test_data_path=input_test_list, image_shape=self.input_shape, image_nums=-1,
            )
            if calc_nums != -1:
                ori_images = ori_images[:calc_nums]
                normed_images = normed_images[:calc_nums]
                image_names = image_names[:calc_nums]

            inference_times = normed_images.shape[0] // batch_size + 1
            for inference_time in range(inference_times):
                this_normed_images = normed_images[
                                     inference_time * batch_size:(inference_time + 1) * batch_size, ...]
                this_ori_images = ori_images[
                                  inference_time * batch_size:(inference_time + 1) * batch_size, ...]
                this_image_names = image_names[
                                   inference_time * batch_size:(inference_time + 1) * batch_size]

                this_pred_mask = self.inference(this_normed_images)
                this_denorm_pred_mask = module.reverse_pred_image(this_pred_mask.numpy())
                if ori_images.shape[0] == 1:
                    this_denorm_pred_mask = np.expand_dims(this_denorm_pred_mask, 0)

                for i in range(this_denorm_pred_mask.shape[0]):
                    bin_denorm_pred_mask = this_denorm_pred_mask[i]
                    this_drawed_image, this_blood_area = module.draw_contours(
                        this_ori_images[i], bin_denorm_pred_mask, dpi=96
                    )
                    cv.imwrite(os.path.join(
                        mask_save_dir, '{}'.format(this_image_names[i])), bin_denorm_pred_mask
                    )
                    cv.imwrite(os.path.join(
                        drawed_save_dir, '{}'.format(this_image_names[i])), this_drawed_image
                    )
        return

    def train(self, start_epoch=1):
        # get training dataset
        train_data = data.get_tfrecord_data(
            self.params.record_dir, self.params.train_record_name,
            self.input_shape, batch_size=self.params.batch_size)
        self.load_model()

        pd_record = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss', 'Time'])
        data_name, original_test_image, norm_test_image = data.get_test_data(
            test_data_path=self.params.test_image_dir, image_shape=self.input_shape, image_nums=-1
        )

        start_time = time.time()
        best_dice = 0.0
        for epoch in range(start_epoch, self.params.epochs):
            for train_image, gt_mask in tqdm.tqdm(
                    train_data, total=self.params.total_samples // self.params.batch_size):
                self.save_ckpt.train_steps.assign_add(1)
                iteration = self.save_ckpt.train_steps.numpy()

                # training step
                train_loss = self.train_step(train_image, gt_mask)
                if iteration % 100 == 0:
                    print('Epoch: {}, Iteration: {}, Loss: {:.2f}, Time: {:.2f} s'.format(
                        epoch, iteration, train_loss, time.time() - start_time))

                    # test step
                    test_pred = self.inference(norm_test_image)
                    module.save_images(
                        image_shape=self.mask_shape, pred=test_pred,
                        save_path=self.pred_test_save_dir, index=iteration, split=False
                    )
                    pd_record = pd_record.append({
                        'Epoch': epoch, 'Iteration': iteration, 'Loss': train_loss.numpy(),
                        'Time': time.time() - start_time}, ignore_index=True
                    )
                    pd_record.to_csv(os.path.join(
                        self.save_dir, '{}_record.csv'.format(self.params.model_name)), index=True, header=True
                    )

            m_dice = self.invalid(epoch)
            if m_dice > best_dice:
                best_dice = m_dice
                print('Best Dice:{}'.format(best_dice))
                self.save_manger.save(checkpoint_number=epoch)
        return

    def invalid(self, epoch):
        invalid_data = data.get_tfrecord_data(
            self.params.record_dir, self.params.invalid_record_name,
            self.input_shape, batch_size=self.params.batch_size, shuffle=False)

        epoch_pred_save_dir = None
        for index, (invalid_image, invalid_mask) in enumerate(
                tqdm.tqdm(invalid_data, total=self.params.invalid_samples // self.params.batch_size + 1)):
            invalid_pred = self.inference(invalid_image)
            epoch_pred_save_dir = os.path.join(self.pred_invalid_save_dir, f'epoch_{epoch}')
            module.save_images(
                image_shape=self.mask_shape, pred=invalid_pred,
                save_path=epoch_pred_save_dir, index=f'{index}', split=False
            )

        # Test model performance
        epoch_cropped_save_dir = os.path.join(
            self.invalid_crop_save_dir, f'epoch_{epoch}'
        )
        utils.crop_image(epoch_pred_save_dir, epoch_cropped_save_dir,
                         self.crop_width, self.crop_height,
                         self.input_shape[0], self.input_shape[1]
                         )
        m_dice, m_iou, m_precision, m_recall = performance.save_performace_to_csv(
            pred_dir=epoch_cropped_save_dir, gt_dir=self.params.gt_mask_dir,
            img_resize=(self.params.height, self.params.width),
            csv_save_name=f'{self.model_name}_epoch_{epoch}',
            csv_save_path=epoch_cropped_save_dir
        )
        return m_dice

