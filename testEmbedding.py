import tensorflow as tf


def test_embedding():
    color_data = {'color': ['R', 'G', 'B', 'A']} 

    color_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    # age_column = tf.feature_column.numeric_column('age',shape=(1,))
    color_embeding = tf.feature_column.embedding_column(color_column, 8)
    color_embeding_dense_tensor = tf.feature_column.input_layer(color_data, [color_embeding])

    embedding = tf.reduce_mean(color_embeding_dense_tensor,1)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print('embeding' + '_' * 40)
        print(session.run([embedding]))

import tensorflow as tf
import numpy as np
_column = ['hist_i','y']
_ValueD = [[''],[0]]

def load_data():
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

load_data()