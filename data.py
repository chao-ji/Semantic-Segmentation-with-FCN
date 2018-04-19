import os

import numpy as np
import tensorflow as tf


#VGG_MEAN = 103.939, 116.779, 123.68 # Blue, Green, Red
VGG_MEAN = 104.00698793, 116.66876762, 122.67891434
VOID_PIXEL = 255
IMAGE_SETS = "ImageSets"
JPEG_IMAGES = "JPEGImages"
SEGMENTATION = "Segmentation"
SEGMENTATION_CLASS = "SegmentationClass"
TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TRAINVAL_TXT = "trainval.txt"
JPG = ".jpg"
PNG = ".png"
DATA_SIZE = 8498
TRAIN = tf.contrib.learn.ModeKeys.TRAIN
INFER = tf.contrib.learn.ModeKeys.INFER


class PascalDataset(object):
  def __init__(self, hparams, mode):
    self._mode = mode

    if mode != INFER:
      (self._initializer, self._labels, self._mask, self._num_valid_pixels,
          self._images) = self._get_iterator(hparams, mode==TRAIN)
    else:
      (self._initializer, self._labels, self._mask, self._num_valid_pixels,
          self._images) = self._get_infer_iterator(hparams)

  @property
  def mode(self):
    return self._mode

  @property
  def labels(self):
    # [1, H, W], dtype=tf.int32
    return self._labels

  @property
  def mask(self):
    # [1, H, W], dtype=tf.float32
    return self._mask

  @property
  def num_valid_pixels(self):
    return self._num_valid_pixels

  @property
  def images(self):
    # [1, H, W, 3], dtype=tf.float32
    return self._images

  def init_iterator(self, sess):
    sess.run(self._initializer)

  def _get_infer_iterator(self, hparams):
    file_dir = hparams.infer_path

    img_filenames = sorted(tf.gfile.Glob(
        os.path.join(file_dir, "images", "*jpg")))
  
    dataset = tf.data.Dataset.from_tensor_slices(img_filenames)

    dataset = dataset.map(lambda fn_img: tf.read_file(fn_img))

    dataset = dataset.map(lambda str_img: tf.image.decode_jpeg(str_img))

    dataset = dataset.map(lambda img: tf.cast(img, tf.float32))

    dataset = dataset.map(lambda img: tf.expand_dims(img, axis=0))

    dataset = dataset.map(lambda img: _subtract_channel_mean(img))

    iterator = dataset.make_one_shot_iterator()

    images = iterator.get_next()

    return None, None, None, None, images


  def _get_iterator(self,
                    hparams,
                    reshuffle_each_iteration=True):
    devkit_path = hparams.devkit_path
    if self.mode == TRAIN:
      fid = os.path.join(devkit_path, IMAGE_SETS, SEGMENTATION, TRAIN_TXT)
    else:
      fid = os.path.join(devkit_path, IMAGE_SETS, SEGMENTATION, VAL_TXT)

    random_seed = hparams.random_seed
    iterations = hparams.iterations

    lbl_filenames = [os.path.join(devkit_path, SEGMENTATION_CLASS,
        (f.strip() + PNG)) for f in tf.gfile.GFile(fid).readlines()]

    img_filenames = [os.path.join(devkit_path, JPEG_IMAGES,
        (f.strip() + JPG)) for f in tf.gfile.GFile(fid).readlines()]

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(lbl_filenames),
        tf.data.Dataset.from_tensor_slices(img_filenames)))

    if self.mode == TRAIN:    
      dataset = dataset.repeat().take(iterations).shuffle(
        DATA_SIZE*10, random_seed, reshuffle_each_iteration)

    dataset = dataset.map(lambda fn_lbl, fn_img:
        (tf.read_file(fn_lbl), tf.read_file(fn_img)))

    dataset = dataset.map(lambda str_lbl, str_img:
        (tf.image.decode_png(str_lbl, channels=3),
        tf.image.decode_jpeg(str_img, channels=3)))

    dataset = dataset.map(lambda lbl, img: (_rgb2index(lbl), img))

    dataset = dataset.map(lambda lbl, img:
        (tf.cast(lbl, tf.int32), tf.cast(img, tf.float32)))

    dataset = dataset.map(lambda lbl, img:
        (tf.expand_dims(lbl, axis=0), tf.expand_dims(img, axis=0)))

    dataset = dataset.map(lambda lbl, img: (lbl, _subtract_channel_mean(img)))

    dataset = dataset.map(lambda lbl, img: (
        lbl,
        tf.where(tf.equal(lbl, VOID_PIXEL),
                 tf.zeros_like(lbl, dtype=tf.float32),
                 tf.ones_like(lbl, dtype=tf.float32)),
        img))

    dataset = dataset.map(lambda lbl, mask, img: (
        tf.where(tf.equal(lbl, VOID_PIXEL),
                 tf.zeros_like(lbl, dtype=tf.int32),
                 lbl),
        mask,
        img))

    dataset = dataset.map(lambda lbl, mask, img: (
        lbl,
        mask,
        tf.reduce_sum(mask),
        img))

    iterator = dataset.make_initializable_iterator()
    labels, mask, num_valid_pixels, images = iterator.get_next()

    return iterator.initializer, labels, mask, num_valid_pixels, images



def _subtract_channel_mean(img):
  img.set_shape([1, None, None, 3])
  r, g, b = tf.unstack(img, axis=3)
  img = tf.stack([r - VGG_MEAN[2], g - VGG_MEAN[1], b - VGG_MEAN[0]], axis=3)
  return img


def _rgb2index(x):
  mask7 = tf.constant(0b10000000, dtype=tf.uint8)
  mask6 = tf.constant(0b01000000, dtype=tf.uint8)
  mask5 = tf.constant(0b00100000, dtype=tf.uint8)
  shift7 = tf.constant(7, dtype=tf.uint8)
  shift6 = tf.constant(6, dtype=tf.uint8)
  shift5 = tf.constant(5, dtype=tf.uint8)
  shift3 = tf.constant(3, dtype=tf.uint8)
  shift2 = tf.constant(2, dtype=tf.uint8)
  shift1 = tf.constant(1, dtype=tf.uint8)

  b0 = tf.bitwise.right_shift(tf.bitwise.bitwise_and(x[:, :, 0], mask7), shift7)
  b1 = tf.bitwise.right_shift(tf.bitwise.bitwise_and(x[:, :, 1], mask7), shift6)
  b2 = tf.bitwise.right_shift(tf.bitwise.bitwise_and(x[:, :, 2], mask7), shift5)
  b3 = tf.bitwise.right_shift(tf.bitwise.bitwise_and(x[:, :, 0], mask6), shift3)
  b4 = tf.bitwise.right_shift(tf.bitwise.bitwise_and(x[:, :, 1], mask6), shift2)
  b5 = tf.bitwise.right_shift(tf.bitwise.bitwise_and(x[:, :, 2], mask6), shift1)
  b6 = tf.bitwise.left_shift(tf.bitwise.bitwise_and(x[:, :, 0], mask5), shift1)
  b7 = tf.bitwise.left_shift(tf.bitwise.bitwise_and(x[:, :, 1], mask5), shift2)
  return b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7
