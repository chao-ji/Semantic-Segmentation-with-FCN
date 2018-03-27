import tensorflow as tf
import numpy as np

import data

from layers import conv_relu_layer
from layers import fc2conv_layer
from layers import project_layer
from layers import upsample_layer

VGG16_CONV_LAYERS = (("conv1_1", "conv1_2"),
                     ("conv2_1", "conv2_2"),
                     ("conv3_1", "conv3_2", "conv3_3"),
                     ("conv4_1", "conv4_2", "conv4_3"),
                     ("conv5_1", "conv5_2", "conv5_3"))


class FCNVGGModel(object):
  def __init__(self, hparams, dataset, mode=None):

    self._upsample_factor = hparams.upsample_factor
    self._mode = mode
    (self._logits, self._loss, self._segmap
        ) = self._build_graph(hparams, dataset)

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
  def loss(self):
    return self._loss

  @property
  def segmap(self):
    return self._segmap

  def _build_graph(self, hparams, dataset):
    ftmps = images = dataset.images
    num_classes = hparams.num_classes
    vgg16_weights = np.load(hparams.vgg16_npy_path).item()
    layers = dict() 

    for b in range(len(VGG16_CONV_LAYERS)):
      for l in range(len(VGG16_CONV_LAYERS[b])):
        name = VGG16_CONV_LAYERS[b][l]
        ftmps = conv_relu_layer(ftmps, vgg16_weights, hparams, name)

      name = "pool%d" % (b + 1)
      ftmps = tf.nn.max_pool(ftmps, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
          padding="SAME", name=name)
      layers[name] = ftmps
      print(name + "\t" + str(ftmps.shape))

    fc6 = fc2conv_layer(ftmps,
                        vgg16_weights=vgg16_weights,
                        hparams=hparams,
                        vgg_layer_name="fc6",
                        vgg_layer_shape=(7, 7, 512, 4096),
                        use_dropout=(self.mode=="train"))

    fc7 = fc2conv_layer(fc6,
                        vgg16_weights=vgg16_weights,
                        hparams=hparams,
                        vgg_layer_name="fc7",
                        vgg_layer_shape=(1, 1, 4096, 4096),
                        use_dropout=(self.mode=="train"))

    out = project_layer(fc7,
                        out_depth=num_classes,
                        hparams=hparams,
                        name="out",
                        zero_initializer=True)

    logits_shape = tf.concat([tf.shape(images)[:3], [num_classes]], axis=0)

    if self.upsample_factor == 32:
      logits = _fcn32s_logits(out, layers, num_classes, hparams, logits_shape)
    elif self.upsample_factor == 16:
      logits = _fcn16s_logits(out, layers, num_classes, hparams, logits_shape)
    elif self.upsample_factor == 8:
      logits = _fcn8s_logits(out, layers, num_classes, hparams, logits_shape)
    else:
      raise ValueError("Invalid upsample factor: %d" % self.upsample_factor)

    loss = _compute_loss(logits, dataset.labels)
    segmap = _compute_segmap(logits)
    return logits, loss, segmap


def _compute_loss(logits, labels):
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.squeeze(labels, squeeze_dims=[3])))
  return loss


def _compute_segmap(logits):
  return tf.expand_dims(
      tf.argmax(logits, axis=3), axis=3)


def _fcn32s_logits(ftmps, layers, num_classes, hparams, logits_shape):
  out_up32x = upsample_layer(ftmps,
                             upsample_factor=32,
                             output_shape=logits_shape,
                             name="out_up32x")
  return out_up32x


def _fcn16s_logits(ftmps, layers, num_classes, hparams, logits_shape):
  pool4_proj = project_layer(layers["pool4"],
                             out_depth=num_classes,
                             hparams=hparams,
                             name="pool4_proj")
  out_up2x = upsample_layer(ftmps,
                            upsample_factor=2,
                            output_shape=tf.shape(pool4_proj),
                            name="out_up2x")

  fused_pool4 = tf.add(pool4_proj, out_up2x, "fused_pool4")

  fused_pool4_up16x = upsample_layer(fused_pool4,
                                     upsample_factor=16,
                                     output_shape=logits_shape,
                                     name="fused_pool4_up16x")

  return fused_pool4_up16x


def _fcn8s_logits(ftmps, layers, num_classes, hparams, logits_shape):
  pool4_proj = project_layer(layers["pool4"],
                             out_depth=num_classes,
                             hparams=hparams,
                             name="pool4_proj")
  pool3_proj = project_layer(layers["pool3"],
                             out_depth=num_classes,
                             hparams=hparams,
                             name="pool3_proj")

  out_up2x = upsample_layer(ftmps,
                            upsample_factor=2,
                            output_shape=tf.shape(pool4_proj),
                            name="out_up2x")

  fused_pool4 = tf.add(pool4_proj, out_up2x, "fused_pool4")

  fused_pool4_up2x = upsample_layer(fused_pool4,
                                    upsample_factor=2,
                                    output_shape=tf.shape(pool3_proj),
                                    name="fused_pool4_up2x")

  fused_pool3 = tf.add(pool3_proj, fused_pool4_up2x, "fused_pool3")

  fused_pool3_up8x = upsample_layer(fused_pool3,
                                    upsample_factor=8,
                                    output_shape=logits_shape,
                                    name="fused_pool3_up8x")

  return fused_pool3_up8x
