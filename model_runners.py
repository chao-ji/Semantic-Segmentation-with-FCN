import tensorflow as tf

import data


class _BaseModelRunner(object):
  mode = None
  def __init__(self, builder, hparams):
    tf.contrib.learn.ModeKeys.validate(type(self).mode)
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._dataset = data.PascalDataset(hparams, type(self).mode)
      self._model = builder(hparams, self.dataset, type(self).mode)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        self._loss = _compute_loss(self.dataset.labels,
                                   self.dataset.mask,
                                   self.dataset.num_valid_pixels,
                                   self.model.logits)
      else:
        self._loss = None

      if type(self).mode == tf.contrib.learn.ModeKeys.TRAIN:
        self._global_step = tf.Variable(0, trainable=False, name="global_step")
        self._learning_rate = self._get_learning_rate(hparams)
        self._update_op = self._get_update_op(hparams)

      self._global_variables_initializer = tf.global_variables_initializer()
      self._saver = tf.train.Saver(tf.global_variables())

#    print("SSSSS")
#    print("labels, " + str(self.dataset.labels.shape) + str(self.dataset.labels.dtype))
#    print("mask, " + str(self.dataset.mask.shape) + str(self.dataset.mask.dtype))
#    print("images, " + str(self.dataset.images.shape) + str(self.dataset.images.dtype))
#    print("logits, " + str(self.model.logits.shape) + str(self.model.logits.dtype))
#    print("predictions, " + str(self.model.predictions.shape) + str(self.model.predictions.dtype))
#    print("EEEEE")

  @property
  def graph(self):
    return self._graph

  @property
  def dataset(self):
    return self._dataset

  @property
  def model(self):
    return self._model

  @property
  def loss(self):
    return self._loss

  def restore_params_from_dir(self, sess, ckpt_dir):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
      print("%s model is loading params from %s..." % (
          type(self).mode.upper(), latest_ckpt))
      self._saver.restore(sess, latest_ckpt)
    else:
      print("%s model is creating fresh params..." % 
          type(self).mode.upper())
      sess.run(self._global_variables_initializer)

  def persist_params_to(self, sess, ckpt):
    print("%s model is saving params to %s..." % (
        type(self).mode.upper(), ckpt))
    self._saver.save(sess, ckpt, global_step=self._global_step)


class FCNVGGModelTrainer(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.TRAIN
  def __init__(self, builder, hparams):
    super(FCNVGGModelTrainer, self).__init__(
        builder=builder, hparams=hparams)

    with self.graph.as_default():
      self.pixel_acc = _pixel_acc(self.dataset.labels,
                                  self.dataset.mask,
                                  self.dataset.num_valid_pixels,
                                  self.model.predictions)
      self.mean_iou = _mean_iou(self.dataset.labels,
                                self.dataset.mask,
                                self.model.predictions,
                                hparams.num_classes)

  def _get_learning_rate(self, hparams):
   return tf.constant(hparams.learning_rate)

  def _get_update_op(self, hparams):
    if hparams.optimizer == "momentum":
      opt = tf.train.MomentumOptimizer(self._learning_rate, hparams.momentum)
    elif hparams.optimizer == "adam":
      opt = tf.train.AdamOptimizer(self._learning_rate)
    else:
      raise ValueError("Unknown optimizer: %s" % hparams.optimizer)
    update_op = opt.minimize(self.loss, global_step=self._global_step)
    return update_op

  def train(self, sess):
    return sess.run([self._update_op,
                     self.loss,
                     self.dataset.labels,
                     self.model.predictions,
                     self.pixel_acc,
                     self.mean_iou,
                     self._global_step])

  def eval_weights(self, sess):
    return sess.run(self.model.weights)


class FCNVGGModelEvaluator(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.EVAL
  def __init__(self, builder, hparams):
    super(FCNVGGModelEvaluator, self).__init__(
        builder=builder, hparams=hparams)

    with self.graph.as_default():
      self.pixel_acc = _pixel_acc(self.dataset.labels,
                                  self.dataset.mask,
                                  self.dataset.num_valid_pixels,
                                  self.model.predictions)
      self.mean_iou = _mean_iou(self.dataset.labels,
                                self.dataset.mask,
                                self.model.predictions,
                                hparams.num_classes)

  def eval(self, sess):
    return sess.run([self.loss,
                     self.dataset.labels,
                     self.model.predictions,
                     self.pixel_acc,
                     self.mean_iou])


class FCNVGGModelInferencer(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.INFER
  def __init__(self, builder, hparams):
    super(FCNVGGModelInferencer, self).__init__(
        builder=builder, hparams=hparams)

  def infer(self, sess):
    return sess.run(self.model.predictions)


def _compute_loss(labels, mask, num_valid_pixels, logits):
  per_pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
#  loss = tf.div(tf.reduce_sum(per_pixel_loss * mask), num_valid_pixels)
  loss = per_pixel_loss * mask
  return loss


def _pixel_acc(labels, mask, num_valid_pixels, predictions):
  pixel_acc = tf.div(tf.reduce_sum(
      tf.cast(tf.equal(labels, predictions), tf.float32) * mask),
      num_valid_pixels)
  return pixel_acc


def _mean_iou(labels, mask, predictions, num_classes):
  labels = tf.reshape(labels, [-1])
  mask = tf.reshape(mask, [-1])
  predictions = tf.reshape(predictions, [-1])

  cm = tf.confusion_matrix(labels, predictions, num_classes, weights=mask)
  axis0_sum = tf.reduce_sum(cm, axis=0) # row
  axis1_sum = tf.reduce_sum(cm, axis=1) # col

  intersection = tf.diag_part(cm)
  union = axis0_sum + axis1_sum - intersection

  num_valid_classes = tf.reduce_sum(tf.cast(tf.not_equal(union, 0), tf.float32))
  union = tf.where(tf.greater(union, 0), union, tf.ones_like(union))
  iou = tf.div(tf.cast(intersection, tf.float32), tf.cast(union, tf.float32))

  return tf.where(tf.greater(num_valid_classes, 0),
      tf.div(tf.reduce_sum(iou), num_valid_classes), 0) 
