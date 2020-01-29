import os
import tensorflow as tf
import pickle
import numpy as np
import re
import tf_metrics
import pprint
import data

os.chdir("D:/gpt-2/src")

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string(
    "data_dir", "D:/TREC/",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir", './output',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "gpt_ckpt", '../model/124M',
    "The directory of gpt checkpoint file.")

flags.DEFINE_string(
    "hparams", '../model/124M',
    "The directory of gpt hparams file.")

flags.DEFINE_integer("class_num", 6, "Total class num.")

flags.DEFINE_integer("maxlen", 130, "Max length.")

flags.DEFINE_integer("batch_count", 8, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 0.0001, "The initial learning rate for Adam.")

flags.DEFINE_integer("epoch_count", 40,
                     "Total number of training epochs to perform.")


flags.DEFINE_integer("train_count", 5452, None)
flags.DEFINE_integer("test_count", 499, None)
flags.DEFINE_bool("do_train", True, None)
flags.DEFINE_bool("do_eval", True, None)
flags.DEFINE_bool("do_predict", True, None)



def predict_labels(estimator, input_fn):
    predictions = estimator.predict(input_fn)
    p = []
    t = []

    for prediction in predictions:
        pred = prediction["pred"]
        true = prediction["true"]
        try:
            p.extend(pred)
            t.extend(true)
        except:
            p.append(pred)
            t.append(true)

    from sklearn.metrics import classification_report

    target_names = ["NUM", "LOC", "HUM", "DESC", "ENTY", "ABBR"]

    report = classification_report(t, p, target_names=target_names)
    print(report)

    from sklearn.metrics import confusion_matrix
    print("----confusion_matrix------")
    print(confusion_matrix(t, p))



def get_loss(features, labels):
    import model
    import json
    hparams = model.default_hparams()
    with open(os.path.join(FLAGS.hparams, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    print(hparams)
    CLASS = FLAGS.class_num
    X = features["input_ids"]
    labels = features["labels"]
    gpt_output = model.model(hparams, X, past=None, scope='model', reuse=False)
    hidden = gpt_output["present"][:,-1,0,-1,:,:]
    print(hidden.shape)
    hidden_flat = tf.reshape(hidden, [hidden.shape[0],-1])
    logits = tf.keras.layers.Dense(CLASS,activation=tf.sigmoid)(hidden_flat)
    pred = tf.argmax(logits,axis=-1)
    label_onehot = tf.one_hot(labels,depth=CLASS, dtype=tf.float32)
    loss_per_example = -tf.reduce_sum(tf.nn.log_softmax(logits) * label_onehot, axis=-1)
    loss = tf.reduce_mean(loss_per_example)
    return loss, pred, labels, loss_per_example





def eval_confusion_matrix(labels, predictions,class_num):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=class_num)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(class_num,class_num), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op



 def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  import collections
  import re
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)
  print("========ckpt vars========")
  pprint.pprint(init_vars)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)







def build_model_fn(num_train_steps):
    def model_fn(features, labels, mode, params):


        loss, pred, true, per_loss = get_loss(features,labels)
        if True and mode == tf.estimator.ModeKeys.TRAIN:
            tvars = tf.trainable_variables()
            print("=======training vars=======")
            pprint.pprint(tvars)
            checkpoint_file = tf.train.get_checkpoint_state(FLAGS.gpt_ckpt).model_checkpoint_path
            variables = tf.global_variables()
            assign, _ = get_assignment_map_from_checkpoint(tvars, checkpoint_file)
            tf.train.init_from_checkpoint(checkpoint_file, assign)
        
        
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            decayed_lr = tf.train.polynomial_decay(FLAGS.learning_rate,
                                        tf.train.get_global_step(), num_train_steps,
                                        0)
            train_op = tf.train.AdamOptimizer(decayed_lr).minimize(loss, global_step=tf.train.get_global_step())
            output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(loss, labels, pred, per_loss):
                predictions = pred
                accuracy = tf.metrics.accuracy(labels, predictions)
                loss_mean = tf.metrics.mean(per_loss)
                precision = tf_metrics.precision(labels, predictions, CLASS, average="macro")
                recall = tf_metrics.recall(labels, predictions, CLASS, average="macro")
                f = tf_metrics.f1(labels, predictions, CLASS, average="macro")
                f_w = tf_metrics.f1(labels, predictions, CLASS, average="weighted")
                cm = eval_confusion_matrix(labels, predictions, CLASS)

                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss_mean,
                    "eval_f1": f,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "cm": cm,
                }

            eval_metrics = (metric_fn, [loss, true, pred])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metric_fn(loss, true, pred, per_loss))
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions={"pred": pred, "true": true,})
        return output_spec

    return model_fn




def main():
    epoch_train_steps = int(FLAGS.train_count / FLAGS.batch_count)
    num_train_steps = epoch_train_steps * float(FLAGS.epoch_count)
    print("Epoch train steps %d" % epoch_train_steps)
    print("Total train steps %d" % num_train_steps)
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=1300, keep_checkpoint_max=2)
    model_fn = build_model_fn(num_train_steps)
    estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.output_dir, config=run_config)

    if FLAGS.do_train:
        train_input_fn = data.build_input_fn(True, "train_X", "train_Y")
        test_input_fn = data.build_input_fn(False, "test_X", "test_Y")

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, throttle_secs=3, steps=None)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_eval:
        test_input_fn = data.build_input_fn(False, "test_X", "test_Y", False)
        estimator.evaluate(test_input_fn)
    if FLAGS.do_predict:
        test_input_fn = data.build_input_fn(False, "test_X", "test_Y", False)
        predict_labels(estimator, test_input_fn)


