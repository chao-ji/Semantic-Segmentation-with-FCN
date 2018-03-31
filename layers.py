import tensorflow as tf
import numpy as np


def conv_relu_layer(ftmps, hparams, weights, name, init_weights):
  kernel_val, bias_val = init_weights[name]
  kernel_initializer = tf.constant_initializer(kernel_val)
  bias_initializer = tf.constant_initializer(bias_val)

  with tf.variable_scope(name):
    kernel = tf.get_variable("kernel",
                             shape=kernel_val.shape,
                             initializer=kernel_initializer,
                             regularizer=_regularizer(hparams))
    bias = tf.get_variable("bias",
                           shape=bias_val.shape,
                           initializer=bias_initializer)
    weights[name] = kernel, bias

    conv = tf.nn.conv2d(ftmps, kernel, (1, 1, 1, 1), "SAME", name="conv")
    relu = tf.nn.relu(conv + bias, name="relu")

  return relu


def fc2conv_layer(
      ftmps, hparams, weights, use_dropout, name, new_shape, init_weights):
  kernel_val, bias_val = init_weights[name]
  kernel_val = kernel_val.reshape(new_shape)
  kernel_initializer = tf.constant_initializer(kernel_val)
  bias_initializer = tf.constant_initializer(bias_val)

  with tf.variable_scope(name):
    kernel = tf.get_variable("kernel",
                             shape=kernel_val.shape,
                             initializer=kernel_initializer,
                             regularizer=_regularizer(hparams))
    bias = tf.get_variable("bias",
                           shape=bias_val.shape,
                           initializer=bias_initializer)
    weights[name] = kernel, bias

    conv = tf.nn.conv2d(ftmps, kernel, (1, 1, 1, 1), "SAME", name="conv")
    relu = tf.nn.relu(conv + bias, name="relu")

    if use_dropout:
      return tf.nn.dropout(relu, hparams.keep_prob, name="dropout")
    else:
      return relu


def project_layer(ftmps, hparams, weights, out_depth, name, init_weights=None):
  in_depth = ftmps.get_shape().as_list()[3]

  if init_weights:
    kernel_val, bias_val = init_weights[name]
    kernel_initializer = tf.constant_initializer(kernel_val)
    bias_initializer = tf.constant_initializer(bias_val)
  else:
    kernel_initializer = tf.zeros_initializer()
    bias_initializer = tf.zeros_initializer()

  with tf.variable_scope(name):
    kernel = tf.get_variable("kernel",
                             shape=(1, 1, in_depth, out_depth),
                             initializer=kernel_initializer,
                             regularizer=_regularizer(hparams))
    bias = tf.get_variable("bias",
                           shape=(out_depth,),
                           initializer=bias_initializer)
    weights[name] = kernel, bias

    conv = tf.nn.conv2d(ftmps, kernel, (1, 1, 1, 1), "SAME", name="conv")
    return conv + bias


def upsample_layer(
    ftmps, weights, upsample_factor, output_shape, name, init_weights=None):

  if init_weights:
    kernel_val = init_weights[name][0]
    kernel_initializer = tf.constant_initializer(kernel_val)
  else:
    depth = ftmps.get_shape().as_list()[3]
    kernel_val = _get_bilinear_weights(upsample_factor, depth)
    kernel_initializer = tf.constant_initializer(kernel_val)

  strides = 1, upsample_factor, upsample_factor, 1

  with tf.variable_scope(name):
    kernel = tf.get_variable("kernel",
                             shape=kernel_val.shape,
                             initializer=kernel_initializer)
    weights[name] = (kernel,)

    conv_transposed = tf.nn.conv2d_transpose(ftmps,
                                             filter=kernel,
                                             output_shape=output_shape,
                                             strides=strides,
                                             padding="SAME",
                                             name=name)
  return conv_transposed


def _get_bilinear_weights(upsample_factor, num_classes):

  kernel_size = upsample_factor * 2 - upsample_factor % 2
  weights_val = np.zeros((kernel_size, kernel_size, num_classes, num_classes),
      dtype=np.float32)

  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5

  og = np.ogrid[:kernel_size, :kernel_size]

  kernel_val = ((1 - abs(og[0] - center) / factor) *
      (1 - abs(og[1] - center) / factor))

  for i in range(num_classes):
    weights_val[:, :, i, i] = kernel_val

  return weights_val


def _regularizer(hparams):
  return tf.contrib.layers.l2_regularizer(scale=hparams.weight_decay)
