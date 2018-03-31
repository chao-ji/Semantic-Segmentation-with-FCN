import tensorflow as tf
import os

VGG_MEAN = 103.939, 116.779, 123.68 # Blue, Green, Red


class FCNVGGDataset(object):
  def __init__(self, hparams, mode):
    self._mode = mode

    if mode != tf.contrib.learn.ModeKeys.INFER: 
      (self._initializer, self._labels, self._images
          ) = self._get_iterator(hparams)
    else:
      (self._initializer, self._labels, self._images
          ) = self._get_infer_iterator(hparams)

  @property
  def mode(self):
    return self._mode

  @property
  def labels(self):
    return self._labels

  @property
  def images(self):
    return self._images

  def init_iterator(self, sess):
    sess.run(self._initializer)

  def _get_infer_iterator(self, hparams):
    file_dir = hparams.infer_dir

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

    return None, None, images

  def _get_iterator(self,
                    hparams,
                    reshuffle_each_iteration=True):
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      file_dir = hparams.train_dir
    else:
      file_dir = hparams.eval_dir

    random_seed = hparams.random_seed
    max_height = hparams.max_height
    max_width = hparams.max_width

    ann_filenames = sorted(tf.gfile.Glob(
        os.path.join(file_dir, "annotations", "*png")))
    ann_basenames = map(
        lambda x: os.path.basename(x).split(".")[0], ann_filenames)

    img_filenames = sorted(tf.gfile.Glob(
        os.path.join(file_dir, "images", "*jpg")))
    img_basenames = map(
        lambda x: os.path.basename(x).split(".")[0], img_filenames)

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(ann_filenames),
        tf.data.Dataset.from_tensor_slices(img_filenames),
        tf.data.Dataset.from_tensor_slices(ann_basenames),
        tf.data.Dataset.from_tensor_slices(img_basenames)))

    dataset = dataset.shuffle(
        1, random_seed, reshuffle_each_iteration)

    dataset = dataset.filter(
        lambda fn_ann, fn_img, bn_ann, bn_img: tf.equal(bn_ann, bn_img))

    dataset = dataset.map(lambda fn_ann, fn_img, bn_ann, bn_img:
        (fn_ann, fn_img))

    dataset = dataset.map(lambda fn_ann, fn_img:
        (tf.read_file(fn_ann), tf.read_file(fn_img)))

    dataset = dataset.map(lambda str_ann, str_img:
        (tf.image.decode_png(str_ann), tf.image.decode_jpeg(str_img)))

    dataset = dataset.filter(lambda ann, img:
        tf.logical_and(tf.equal(tf.shape(ann)[2], 1),
            tf.equal(tf.shape(img)[2], 3)))

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      dataset = dataset.map(
          lambda ann, img: tf.concat([ann, img], axis=2))
      dataset = dataset.map(
          lambda stacked: _crop_to_size(stacked, max_height, max_width))
      dataset = dataset.map(
          lambda stacked: tuple(tf.split(stacked, [1, 3], 2)))

    dataset = dataset.map(lambda ann, img:
        (tf.cast(ann, tf.int32), tf.cast(img, tf.float32)))

    dataset = dataset.map(lambda ann, img: 
        (tf.expand_dims(ann, axis=0), tf.expand_dims(img, axis=0)))

    dataset = dataset.map(lambda ann, img: (ann, _subtract_channel_mean(img)))
   
    iterator = dataset.make_initializable_iterator()
    labels, images = iterator.get_next()
    labels.set_shape([1, None, None, 1])
    return iterator.initializer, labels, images


def _crop_to_size(x, h0, w0):
  shape = tf.shape(x)
  h, w, d = shape[0], shape[1], shape[2]
  cropped = tf.cond(h > h0,
      lambda: tf.cond(w > w0,
          lambda: tf.random_crop(x, tf.concat([[h0], [w0], [d]], axis=0)),
          lambda: tf.random_crop(x, tf.concat([[h0], [w],  [d]], axis=0))),
      lambda: tf.cond(w > w0,
          lambda: tf.random_crop(x, tf.concat([[h],  [w0], [d]], axis=0)),
          lambda: x))
  return cropped


def _subtract_channel_mean(img):
  img.set_shape([1, None, None, 3])
  r, g, b = tf.unstack(img, axis=3)
  img = tf.stack([r - VGG_MEAN[2], g - VGG_MEAN[1], b - VGG_MEAN[0]], axis=3)
  return img
