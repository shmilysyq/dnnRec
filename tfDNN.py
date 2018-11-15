from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf
import math

embed_size = 64
n_classes = len(itemIdx)
X_FEATURE = 'x'  # Name of the input feature.

def my_model(features, labels, mode, params):
  """DNN with three hidden layers."""
  # Create three fully connected layers respectively of size 10, 20, and 10.
  net = features[X_FEATURE]
  # Use `input_layer` to apply the feature columns.
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  # concat feature
  
  for units in [128, 256, 128]:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

  # Compute loss.
  # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([n_classes, units],
                            stddev=1.0 / math.sqrt(units)))
    nce_biases = tf.Variable(tf.zeros([n_classes]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                        biases=nce_biases,
                        labels=labels,
                        inputs=net,
                        num_sampled=10,
                        num_classes=n_classes))

# Compute logits (1 per class).
  logits = tf.layers.dense(net, 3, activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class': predicted_classes,
        'prob': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  # Create training op with exponentially decaying learning rate.
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.1, global_step=global_step,
        decay_steps=100, decay_rate=0.001)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  # Compute evaluation metrics.
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=eval_metric_ops)

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)


    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def main(unused_argv):
  yanxuan = load()
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      yanxuan.data, yanxuan.target, test_size=0.2, random_state=42)
  
  my_feature_columns = []
  for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


  classifier = tf.estimator.Estimator(
      model_fn=my_model,
      params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,})

  # Train.
  classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
  eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

if __name__ == '__main__':
  tf.app.run()
