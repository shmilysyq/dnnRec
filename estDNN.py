from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import shutil
import sys

import tensorflow as tf

_CSV_COLUMNS = ['hist_i','target']

_CSV_COLUMN_DEFAULTS = [[''],['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

n_class = 1

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='./model/census_model',
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='./data/train.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='./data/test.csv',
    help='Path to the test data.')

def build_estimator(model_dir):
    columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.Estimator(model_fn=my_model,config=run_config,params={"feature_columns":columns,"hidden_units":hidden_units,'n_class':n_class})


def my_model(features, labels, mode, params):    

    # Create three fully connected layers respectively of size 10, 20, and 10.
    # Use `input_layer` to apply the feature columns.
    feature_columns = params["feature_columns"]
    net = tf.feature_column.input_layer(features, feature_columns)
    # embeding_dense_tensor = tf.feature_column.input_layer(features, [feature_columns['hist_i']])
    ## avg
    embedding = tf.reduce_mean(net,1)

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_class'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def build_model_columns():
    """Builds a set of feature columns."""
    # Continuous columns
    # age = tf.feature_column.numeric_column('age')

    # # Transformations.
    # age_buckets = tf.feature_column.bucketized_column(
    #     age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    itemID = tf.feature_column.categorical_column_with_vocabulary_list('hist_i',vocabulary_list=['1','2','3','4','5'],default_value=-1,dtype=tf.string)

    columns = [
        # To show an example of embedding
        tf.feature_column.embedding_column(itemID, dimension=64)
    ]
    return columns

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        print(value)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('target')
        print(features,labels)
        return features, labels

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
    # Clean up the model directory if present
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
