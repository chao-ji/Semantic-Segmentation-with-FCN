from hp import *
import model
import data
import model_runners
import os
import numpy as np
import datetime

tf.logging.set_verbosity(tf.logging.WARN)


ckpt_dir = "/tmp/fcn_8s/"
hparams.upsample_factor = 8

builder = model.FCNVGGModel
print("Creating trainer...")
trainer = model_runners.FCNVGGModelTrainer(builder, hparams)
print("Done creating trainer.\n")

print("Creating evaluator...")
evaluator = model_runners.FCNVGGModelEvaluator(builder, hparams)
print("Done creating evaluator.\n")

print("\n")
train_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=trainer.graph)
print("\n")
eval_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=evaluator.graph)
print("\n")

trainer.restore_params_from_dir(train_sess, ckpt_dir)


def evaluate(evaluator, eval_sess):
  buf = []
  evaluator.dataset.init_iterator(eval_sess)
  while True:
    try:
      loss, acc, iou, dl, rl = evaluator.eval(eval_sess)
    except tf.errors.OutOfRangeError:
      break
    buf.append((loss.mean(), acc, iou, dl.mean(), rl.mean()))
  buf = np.array(buf)
  print("Num of validation images: %d" % buf.shape[0])
  return buf.mean(axis=0)

summary_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, "train_log"), trainer.graph)

buf = []

trainer.dataset.init_iterator(train_sess)

while True:
  try:
    _, loss, acc, iou, global_step, summary, dl, rl = trainer.train(train_sess)
  except tf.errors.OutOfRangeError:
    break

  buf.append((loss.mean(), acc, iou, dl.mean(), rl.mean()))
  summary_writer.add_summary(summary, global_step)

  if global_step == 1:
    print("%d, loss: %f, acc: %f, iou: %f, dl: %f, rl: %f" % (0, loss.mean(), acc, iou, dl.mean(), rl.mean()))
  elif len(buf) == 200:
    buf = np.array(buf)
    print("%d, loss: %f, acc: %f, iou: %f, dl: %f, rl: %f" % (global_step, np.mean(buf[:, 0]),
                                                             np.mean(buf[:, 1]),
                                                             np.mean(buf[:, 2]),
                                                             np.mean(buf[:, 3]),
                                                             np.mean(buf[:, 4])))
    buf = []

  if global_step % 10000 == 0:
    trainer.persist_params_to(train_sess, os.path.join(ckpt_dir, "fcn"))

    evaluator.restore_params_from_dir(eval_sess, ckpt_dir)
    result = evaluate(evaluator, eval_sess)
    print("Validation loss: %f, acc: %f, iou: %f, dl: %f, rl: %f" % (result[0], result[1], result[2], result[3], result[4]))
    print(datetime.datetime.now())
    print
