import tensorflow as tf
import numpy as np

import data
import collections

from layers import conv_relu_layer
from layers import fc2conv_layer
from layers import project_layer
from layers import upsample_layer

VGG16_CONV_LAYERS = (("conv1_1", "conv1_2"),
                     ("conv2_1", "conv2_2"),
                     ("conv3_1", "conv3_2", "conv3_3"),
                     ("conv4_1", "conv4_2", "conv4_3"),
                     ("conv5_1", "conv5_2", "conv5_3"))

UPSAMPLE_FACTORS = 32, 16, 8

TRAIN = tf.contrib.learn.ModeKeys.TRAIN
INFER = tf.contrib.learn.ModeKeys.INFER


class FCNVGGModel(object):
  def __init__(self, hparams, dataset, mode=None):
    if hparams.upsample_factor not in UPSAMPLE_FACTORS:
      raise ValueError("Invalid upsample factor: %d" % hparams.upsample_factor)

    self._upsample_factor = hparams.upsample_factor
    self._mode = mode
    self._weights = collections.OrderedDict()
    self._logits, self._predictions = self._build_graph(hparams, dataset)

  @property
  def upsample_factor(self):
    return self._upsample_factor

  @property
  def mode(self):
    return self._mode

  @property
  def logits(self):
    return self._logits

  @property
  def predictions(self):
    return self._predictions

  @property
  def weights(self):
    return self._weights

  def _build_graph(self, hparams, dataset):
    ftmps = images = dataset.images
    num_classes = hparams.num_classes
    init_weights = _get_init_weights(hparams)
    weights = self.weights

    layers = collections.OrderedDict() 

    for b in range(len(VGG16_CONV_LAYERS)):
      for l in range(len(VGG16_CONV_LAYERS[b])):
        name = VGG16_CONV_LAYERS[b][l]
        ftmps = conv_relu_layer(ftmps, hparams, weights, name, init_weights)

      name = "pool%d" % (b + 1)
      ftmps = tf.nn.max_pool(ftmps, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
          padding="SAME", name=name)
      layers[name] = ftmps
      print(name + "\t" + str(ftmps.shape))

    fc6 = fc2conv_layer(ftmps,
                        hparams=hparams,
                        weights=weights,
                        use_dropout=(self.mode==TRAIN),
                        name="fc6",
                        new_shape=(7, 7, 512, 4096),
                        init_weights=init_weights)

    fc7 = fc2conv_layer(fc6,
                        hparams=hparams,
                        weights=weights,
                        use_dropout=(self.mode==TRAIN),
                        name="fc7",
                        new_shape=(1, 1, 4096, 4096),
                        init_weights=init_weights)

    out = project_layer(fc7,
                        hparams=hparams,
                        weights=weights,
                        out_depth=num_classes,
                        name="out",
                        init_weights=init_weights)

    logits_shape = tf.concat([tf.shape(images)[:3], [num_classes]], axis=0)

    if self.upsample_factor == 32:
      logits = _fcn32s_logits(out,
          hparams, weights, layers, num_classes, logits_shape, init_weights)
    elif self.upsample_factor == 16:
      logits = _fcn16s_logits(out,
          hparams, weights, layers, num_classes, logits_shape, init_weights)
    elif self.upsample_factor == 8:
      logits = _fcn8s_logits(out,
          hparams, weights, layers, num_classes, logits_shape, init_weights)

    predictions = tf.argmax(logits, axis=3, output_type=tf.int32) 
    return logits, predictions


def _get_init_weights(hparams):
  if hparams.upsample_factor == 32:
    path = hparams.vgg16_npy_path
  elif hparams.upsample_factor == 16:
    path = hparams.fcn_32s_npy_path
  elif hparams.upsample_factor == 8:
    if hparams.fcn8s_oneshot:
      path = hparams.vgg16_npy_path
    else:
      path = hparams.fcn_16s_npy_path
  init_weights = np.load(path).item()
  return init_weights


def _fcn32s_logits(
    ftmps, hparams, weights, layers, num_classes, logits_shape, init_weights):
  out_up32x = upsample_layer(ftmps,
                             hparams,
                             weights=weights,
                             upsample_factor=32,
                             output_shape=logits_shape,
                             name="out_up32x",
                             init_weights=init_weights)
  return out_up32x


def _fcn16s_logits(
    ftmps, hparams, weights, layers, num_classes, logits_shape, init_weights):
  pool4_proj = project_layer(layers["pool4"],
                             hparams=hparams,
                             weights=weights,
                             out_depth=num_classes,
                             name="pool4_proj",
                             init_weights=init_weights)
  out_up2x = upsample_layer(ftmps,
                            hparams,
                            weights=weights,
                            upsample_factor=2,
                            output_shape=tf.shape(pool4_proj),
                            name="out_up2x",
                            init_weights=init_weights)

  fused_pool4 = tf.add(pool4_proj, out_up2x, "fused_pool4")

  fused_pool4_up16x = upsample_layer(fused_pool4,
                                     hparams,
                                     weights=weights,
                                     upsample_factor=16,
                                     output_shape=logits_shape,
                                     name="fused_pool4_up16x",
                                     init_weights=init_weights)
  return fused_pool4_up16x


def _fcn8s_logits(
    ftmps, hparams, weights, layers, num_classes, logits_shape, init_weights):
  pool4 = layers["pool4"]
  if hparams.fcn8s_oneshot == True:
    pool4 = tf.multiply(pool4, 0.01)
  pool4_proj = project_layer(pool4,
                             hparams=hparams,
                             weights=weights,
                             out_depth=num_classes,
                             name="pool4_proj",
                             init_weights=init_weights)

  pool3 = layers["pool3"]
  if hparams.fcn8s_oneshot == True:
    pool3 = tf.multiply(pool3, 0.0001)
  pool3_proj = project_layer(pool3,
                             hparams=hparams,
                             weights=weights,
                             out_depth=num_classes,
                             name="pool3_proj",
                             init_weights=init_weights)

  out_up2x = upsample_layer(ftmps,
                            hparams,
                            weights=weights,
                            upsample_factor=2,
                            output_shape=tf.shape(pool4_proj),
                            name="out_up2x",
                            init_weights=init_weights)

  fused_pool4 = tf.add(pool4_proj, out_up2x, "fused_pool4")

  fused_pool4_up2x = upsample_layer(fused_pool4,
                                    hparams,
                                    weights=weights,
                                    upsample_factor=2,
                                    output_shape=tf.shape(pool3_proj),
                                    name="fused_pool4_up2x",
                                    init_weights=init_weights)

  fused_pool3 = tf.add(pool3_proj, fused_pool4_up2x, "fused_pool3")

  fused_pool3_up8x = upsample_layer(fused_pool3,
                                    hparams,
                                    weights=weights,
                                    upsample_factor=8,
                                    output_shape=logits_shape,
                                    name="fused_pool3_up8x",
                                    init_weights=init_weights)
  return fused_pool3_up8x
