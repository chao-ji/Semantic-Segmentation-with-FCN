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
        self._loss, self._pred_loss, self._reg_loss = _compute_loss(self.dataset.labels,
                                   self.dataset.mask,
                                   self.model.logits)
      else:
        self._loss = None

      if type(self).mode == tf.contrib.learn.ModeKeys.TRAIN:
        self._global_step = tf.Variable(0, trainable=False, name="global_step")
        self._learning_rate = self._get_learning_rate(hparams)
        self._update_op = self._get_update_op(hparams)

      self._global_variables_initializer = tf.global_variables_initializer()
      self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)


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
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("total loss", tf.reduce_mean(self.loss)),
          tf.summary.scalar("pred loss", tf.reduce_mean(self._pred_loss)),
          tf.summary.scalar("reg loss", tf.reduce_mean(self._reg_loss)),
          tf.summary.scalar("pixel acc", self.pixel_acc),
          tf.summary.scalar("meanIU", self.mean_iou)])

  def _get_learning_rate(self, hparams):
   return tf.constant(hparams.learning_rate)

  def _get_update_op(self, hparams):
    if hparams.optimizer == "momentum":
      opt = tf.train.MomentumOptimizer(self._learning_rate, hparams.momentum)
    elif hparams.optimizer == "adam":
      opt = tf.train.AdamOptimizer(self._learning_rate)
    else:
      raise ValueError("Unknown optimizer: %s" % hparams.optimizer)

    if hparams.bias_lr_multiplier is not None:
      mult = hparams.bias_lr_multiplier
      vs = tf.trainable_variables()
      non_biases = [v for v in vs if "bias" not in v.name]
      biases = [v for v in vs if "bias" in v.name]
  
      gvs_non_biases = opt.compute_gradients(self._loss, non_biases)
      gvs_biases = [(g * mult, v)
          for g, v in opt.compute_gradients(self._loss, biases)]
      grads_and_vars = gvs_non_biases + gvs_biases
      update_op = opt.apply_gradients(grads_and_vars, self._global_step)
    else:
      print("same lr")
      update_op = opt.minimize(self._loss, global_step=self._global_step)

    return update_op

  def train(self, sess):
    return sess.run([self._update_op,
                     self.loss,
                     self.pixel_acc,
                     self.mean_iou,
                     self._global_step,
                     self.train_summary,
                     self._pred_loss,
                     self._reg_loss,
                     self.dataset.num_valid_pixels])

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
                     self.pixel_acc,
                     self.mean_iou,
                     self._pred_loss,
                     self._reg_loss,
                     self.dataset.num_valid_pixels])


class FCNVGGModelInferencer(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.INFER
  def __init__(self, builder, hparams):
    super(FCNVGGModelInferencer, self).__init__(
        builder=builder, hparams=hparams)

  def infer(self, sess):
    return sess.run(self.model.predictions)


def _compute_loss(labels, mask, logits):
  per_pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)

  pred_loss = per_pixel_loss * mask

  reg_loss = tf.add_n(
      tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

  total_loss = pred_loss + reg_loss
  return total_loss, pred_loss, reg_loss


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
