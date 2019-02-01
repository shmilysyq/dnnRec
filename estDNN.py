from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import shutil
import sys
from utils.data_process import * 

_CSV_COLUMNS = ['click_hist_i','click_hist_c','click_last_i','click_last_c','y']

_SEQ_COLUMNS = ['click_hist_i','click_hist_c']

_CSV_COLUMN_DEFAULTS = [[''],[''],[''],[''],[0]]

_NUM_EXAMPLES = {
    'train': 2,
    'validation': 3,
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
    hidden_units = [100, 75, 50, 25]
    
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.Estimator(model_fn=my_model,config=run_config,params={"hidden_units":hidden_units,'n_class':n_class})


def my_model(features, labels, mode, params):    

    # Create three fully connected layers respectively of size 10, 20, and 10.
    # Use `input_layer` to apply the feature columns.
    net = build_model_features(features)

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
    # Compute loss.
    # print(labels)
    # labels = tf.reshape(labels,[-1,1])
    # print(labels)
    # weights = tf.get_variable("nce_weight",shape=[n_class, units])
    # biases = tf.get_variable("nce_biase",shape=[n_class])
    # loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights,
    #                  biases=biases,
    #                  labels=labels,
    #                  inputs=net,
    #                  num_sampled=10,
    #                  num_classes=n_class))
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

def build_model_features(raw_features):
    """Builds a set of feature columns."""
    features = {}
    features.update(raw_features)

    # click_item_h & last_click_item  share embedding
    itemIDCol = tf.feature_column.categorical_column_with_vocabulary_file("click_hist_i","./data/item_meta")
    itemIDCol_1 = tf.feature_column.categorical_column_with_vocabulary_file("click_last_i","./data/item_meta")

    itemEmbed = tf.feature_column.shared_embedding_columns([itemIDCol, itemIDCol_1], 64 , combiner='mean')
    clickItemListFeature = tf.feature_column.input_layer(raw_features, itemEmbed[0])
    features.update({"click_hist_i":clickItemListFeature})
    lastItemFeature = tf.feature_column.input_layer(raw_features, itemEmbed[1])
    features.update({"click_last_i":lastItemFeature})


    # click_cat_h & last_click_cat share embedding
    catIDCol = tf.feature_column.categorical_column_with_vocabulary_file("click_hist_c","./data/cat_meta")
    catIDCol_1 = tf.feature_column.categorical_column_with_vocabulary_file("click_last_c","./data/cat_meta")
    catEmbed = tf.feature_column.shared_embedding_columns([catIDCol,catIDCol_1],64)
    clickCatListFeature = tf.feature_column.input_layer(raw_features, catEmbed[0])
    features.update({"click_hist_c":clickCatListFeature})
    lastCatFeature = tf.feature_column.input_layer(raw_features, catEmbed[1])
    features.update({"click_last_c":lastCatFeature})

    return features

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('y')
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file).skip(1)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    process_features(features,_SEQ_COLUMNS)
    print("precess feature "+"_"*40)
    print(features)
    print("label "+"_"*40)
    print(labels)

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
