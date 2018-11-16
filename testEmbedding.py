import tensorflow as tf


def test_embedding():
    color_data = {'color': [['R', 'G', 'B', 'A']],'age':[1]} 

    color_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    age_column = tf.feature_column.numeric_column('age')
    color_embeding = tf.feature_column.embedding_column(color_column, 8)
    net = tf.feature_column.input_layer(color_data, [age_column,color_embeding])

    # embedding = tf.reduce_mean(color_embeding_dense_tensor,1)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print('embeding' + '_' * 40)
        print(session.run([net]))

test_embedding()

import tensorflow as tf
import numpy as np
_column = ['hist_i','y']
_ValueD = [[''],[0]]

def load_data():
    ds = tf.data.TextLineDataset("./data/train.csv").skip(1)

    COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']
    FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]
    def _parse_line(line):
        # Decode the line into its fields
        fields = tf.decode_csv(line, FIELD_DEFAULTS)

        # Pack the result into a dictionary
        features = dict(zip(COLUMNS,fields))

        # Separate the label from the features
        label = features.pop('label')

        return features, label

    ds = ds.map(_parse_line)
    print(ds)
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())

    #     session.run(tf.tables_initializer())

    #     print(session.run([ds]))
# load_data()