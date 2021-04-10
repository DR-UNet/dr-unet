# DR-UNet: A robust deep learning segmentation method for hematoma volumetric detection in intracerebral hemorrhage

------


## Getting Started

- [data.py](drunet/data.py)  Used to make your own dataset. Making your own dataset needs to satisfy having original images and the ground truth images. The completed dataset is a unique data format of tensorflow. 

  ```python
  from data import make_data
  
  # make tfrecords datasets
  make_data(image_shape, image_dir, mask_dir, out_name, out_dir)
  
  # get tfrecords datasets
  dataset = get_tfrecord_data(
      tf_record_path, tf_record_name, data_shape, batch_size=32, repeat=1, shuffle=True)
  ```

- [loss.py](drunet/loss.py)  According to the characteristics of the cerebral hematoma dataset, in order to obtain higher segmentation accuracy. We use binary cross entropy with dice as the loss function of DR-UNet.

- [module.py](drunet/module.py) This file contains several auxiliary functions for image processing.

- [utils.py](drunet/utils.py) This python file contains several auxiliary functions for file operations.

- [performance.py](drunet/performance.py) In order to evaluate the segmentation performance of the model, this file contains auxiliary functions for the calculation of several common segmentation indicators.

- [drunet.py](drunet/model/dr_unet.py) This file contains the specific implementation of DR-UNet and three reduced dimensional residual convolution units (RDRCUs).

  ```python
  from model import dr_unet
  
  model = dr_unet.dr_unet(input_shape=(256, 256, 1))
  model.summary()
  ```

- [segment.py](drunet/segment.py) This file shows how to train, test and verification DR-UNet on your own dataset. Including hematoma segmentation and hematoma volume estimation.

  ```python
  import pathlib
  
  # Parameter configuration
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
  parser.add_argument('--train-record-name', type=str, default=r'train_data', 
                      help='the train record save name')
  parser.add_argument('--test-image-dir', default=r'', type=str, 
                      help='the path of test images dir')
  parser.add_argument('--invalid-record-name', type=str, default=r'test_data', 
                      help='the invalid record save name')
  parser.add_argument('--gt-mask-dir', default=r'', type=str, 
                      help='the ground truth dir of validation set')
  parser.add_argument('--invalid-volume-dir', default=r'', type=str, 
                      help='estimation bleeding volume')
  args = parser.parse_args()
  
  
  segment = Segmentation(args)
  # start training
  segment.train() 
  # predict hematoma volume
  segment.predict_blood_volume(input_dir, save_dir, calc_nums=-1, dpi=96, thickness=0.45)
  ```

  [train_segment.py](drunet/train_segment.py) If you want to train the segmentation model, you can run this file directly after filling in the data path.
  
  ```python
  import segment
  
  if __name__ == '__main__':
      Seg = segment.Segmentation()
      # start training
      Seg.train()
  ```
  
  [predict_segment.py](drunet/predict_segment.py) If you want to predict the segmentation result, you can run this file directly after filling in the ct images path.
  
  ```python
  import segment
  
  if __name__ == '__main__':
      Seg = segment.Segmentation()
      # start predict
      input_dir = r''  # fill in the image path
      save_dir = r''  # fill in predict results save path
      Seg.predict_and_save(input_dir, save_dir)
  
  ```
  
  [predict_volume.py](drunet/predict_volume.py) If you want to predict the complete hematoma volume of a patient, after filling in the path, then you can run this file,
  
  ```python
  import segment
  
  if __name__ == '__main__':
      Seg = segment.Segmentation()
      # start predict
      input_dir = r''  # fill in the image path
      save_dir = r''  # fill in save path
      Seg.predict_blood_volume(input_dir, save_dir, thickness=0.45)
  ```
  
  [test_performance.py](drunet/test_performance.py) If you want to understand the segmentation performance of the model, you need to fill in the relevant path first, and then run this file.
  
  ```python
  import performance
  
  if __name__ == '__main__':
      # test model segmentation performance
      pred_path = r''  # predict result path
      gt_path = r''  # ground truth path
      calc_performance(pred_path, gt_path, img_resize=(1400, 1400))
  ```
  
  

## Requirements

Python 3.6, TensorFlow 2.1 and other common packages listed in `requirements.txt`.

