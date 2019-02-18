from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import shutil
import sys
from utils.data_process import * 

_CSV_COLUMNS = ['item_hist','cat1_session','cat2_session','cat3_session','cat4_session','click_last_i','click_last_cat1','click_last_cat2','click_last_cat3','click_last_cat4','label','user_id','item_id','cat1','cat2','cat3','cat4']

_SEQ_COLUMNS = ['item_hist','cat1_session','cat2_session','cat3_session','cat4_session']

_CSV_COLUMN_DEFAULTS = [[''],[''],[''],[''],[''],[''],[''],[''],[''],[''],[0],[''],[''],[''],[''],[''],['']]

_NUM_EXAMPLES = {
    'train': 6227065,
    'validation': 890119,
}

n_class = 15000

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='./model/dnn_model',
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=4, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=1024, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='./data/train_sample_test',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='./data/test_sample_test',
    help='Path to the test data.')

def build_estimator(model_dir):
    hidden_units = [1024,512,128]
    
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.Estimator(model_fn=my_model,config=run_config,params={"hidden_units":hidden_units,'n_class':n_class})


def my_model(features, labels, mode, params):    

    n_class = params['n_class']
    # Create three fully connected layers respectively of size 10, 20, and 10.
    # Use `input_layer` to apply the feature columns.
    net = build_model_features(features)
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute loss.
    weights = tf.get_variable("nce_weight",shape=[n_class, 128])
    biases = tf.get_variable("nce_biase",shape=[n_class])
    labels_a = tf.expand_dims(labels,1)
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights,biases,labels_a,net,20, n_class))


    # Compute logits (1 per class).
    # logits = tf.layers.dense(net, params['n_class'], activation=None,kernel_constraint=weights,bias_constraint=biases,trainable=False)
    user_vec = tf.expand_dims(net,1)
    logits = tf.matmul(net, weights,transpose_b=True) + biases
    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    y_hat = tf.nn.softmax(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    labels_a_cast = tf.cast(labels_a,dtype=tf.int64)
    precision_k = tf.metrics.average_precision_at_k(labels=labels_a_cast,
                                   predictions=y_hat,k=10,
                                   name='precision_10')
    # auc = tf.metrics.auc(labels=labels,predictions=predicted_classes,name= 'auc')
    metrics = {'precision_k':precision_k}
    tf.summary.scalar('precision_k', precision_k[1])
    # tf.summary.scalar('auc', auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def build_model_features(raw_features):

    """Builds a set of feature columns."""
    features = {}
    features.update(raw_features)

    # click_item_h & last_click_item  share embedding
    # targetItemCol = tf.feature_column.categorical_column_with_vocabulary_file("item_id","./data/item_meta")
    itemIDCol = tf.feature_column.categorical_column_with_vocabulary_file("item_hist","./data/item_meta")
    itemIDCol_1 = tf.feature_column.categorical_column_with_vocabulary_file("click_last_i","./data/item_meta")

    itemEmbed = tf.feature_column.shared_embedding_columns([itemIDCol, itemIDCol_1], 128 , combiner='mean')
    clickItemListFeature = tf.feature_column.input_layer(raw_features, itemEmbed[0])
    features.update({"item_hist":clickItemListFeature})
    lastItemFeature = tf.feature_column.input_layer(raw_features, itemEmbed[1])
    features.update({"click_last_i":lastItemFeature})

    # targetItemFeature = tf.feature_column.input_layer(raw_features, itemEmbed[2])
    # features.update({"item_id":targetItemFeature})


    # click_cat_h & last_click_cat share embedding
    targetCat1Col = tf.feature_column.categorical_column_with_vocabulary_file("cat1","./data/cat1_meta")
    cat1IDCol = tf.feature_column.categorical_column_with_vocabulary_file("cat1_session","./data/cat1_meta")
    cat1IDCol_1 = tf.feature_column.categorical_column_with_vocabulary_file("click_last_cat1","./data/cat1_meta")
    cat1Embed = tf.feature_column.shared_embedding_columns([cat1IDCol,cat1IDCol_1,targetCat1Col],128)
    clickCat1ListFeature = tf.feature_column.input_layer(raw_features, cat1Embed[0])
    features.update({"cat1_session":clickCat1ListFeature})
    lastCat1Feature = tf.feature_column.input_layer(raw_features, cat1Embed[1])
    features.update({"click_last_cat1":lastCat1Feature})

    targetcat1Feature = tf.feature_column.input_layer(raw_features, cat1Embed[2])
    features.update({"cat1":targetcat1Feature})

    # click_cat_h & last_click_cat share embedding
    targetCat2Col = tf.feature_column.categorical_column_with_vocabulary_file("cat2","./data/cat2_meta")
    cat2IDCol = tf.feature_column.categorical_column_with_vocabulary_file("cat2_session","./data/cat2_meta")
    cat2IDCol_1 = tf.feature_column.categorical_column_with_vocabulary_file("click_last_cat2","./data/cat2_meta")
    cat2Embed = tf.feature_column.shared_embedding_columns([cat2IDCol,cat2IDCol_1,targetCat2Col],128)
    clickCat2ListFeature = tf.feature_column.input_layer(raw_features, cat2Embed[0])
    features.update({"cat2_session":clickCat2ListFeature})
    lastCat2Feature = tf.feature_column.input_layer(raw_features, cat2Embed[1])
    features.update({"click_last_cat2":lastCat2Feature})
    targetcat2Feature = tf.feature_column.input_layer(raw_features, cat2Embed[2])
    features.update({"cat2":targetcat2Feature})


    # click_cat_h & last_click_cat share embedding
    targetCat3Col = tf.feature_column.categorical_column_with_vocabulary_file("cat3","./data/cat3_meta")
    cat3IDCol = tf.feature_column.categorical_column_with_vocabulary_file("cat3_session","./data/cat3_meta")
    cat3IDCol_1 = tf.feature_column.categorical_column_with_vocabulary_file("click_last_cat3","./data/cat3_meta")
    cat3Embed = tf.feature_column.shared_embedding_columns([cat3IDCol,cat3IDCol_1,targetCat3Col],128)
    clickCat3ListFeature = tf.feature_column.input_layer(raw_features, cat3Embed[0])
    features.update({"cat3_session":clickCat3ListFeature})
    lastCat3Feature = tf.feature_column.input_layer(raw_features, cat3Embed[1])
    features.update({"click_last_cat3":lastCat3Feature})

    targetcat3Feature = tf.feature_column.input_layer(raw_features, cat3Embed[2])
    features.update({"cat3":targetcat3Feature})

    # click_cat_h & last_click_cat share embedding
    targetCat4Col = tf.feature_column.categorical_column_with_vocabulary_file("cat4","./data/cat4_meta")
    cat4IDCol = tf.feature_column.categorical_column_with_vocabulary_file("cat4_session","./data/cat4_meta")
    cat4IDCol_1 = tf.feature_column.categorical_column_with_vocabulary_file("click_last_cat4","./data/cat4_meta")
    cat4Embed = tf.feature_column.shared_embedding_columns([cat4IDCol,cat4IDCol_1,targetCat4Col],128)
    clickCat4ListFeature = tf.feature_column.input_layer(raw_features, cat4Embed[0])
    features.update({"cat4_session":clickCat4ListFeature})
    lastCat4Feature = tf.feature_column.input_layer(raw_features, cat4Embed[1])
    features.update({"click_last_cat4":lastCat4Feature})

    targetcat4Feature = tf.feature_column.input_layer(raw_features, cat4Embed[2])
    features.update({"cat4":targetcat4Feature})

    input_layer = tf.concat([clickItemListFeature,lastItemFeature,clickCat1ListFeature,lastCat1Feature,clickCat2ListFeature,lastCat2Feature,clickCat3ListFeature,lastCat3Feature,clickCat4ListFeature,lastCat4Feature],1)
    
    # item_vec = tf.concat([targetItemFeature,targetcat1Feature,targetcat2Feature,targetcat3Feature,targetcat4Feature],1)
    return input_layer

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        # print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
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

    process_features(features,_SEQ_COLUMNS)
    # print("precess feature "+"_"*40)
    # print(features)
    # print("label "+"_"*40)
    # print(labels)

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
        
        ## predict metric
        # predictions = model.predict(input_fn=lambda: input_fn(
        #     FLAGS.test_data, 1, False, FLAGS.batch_size))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
