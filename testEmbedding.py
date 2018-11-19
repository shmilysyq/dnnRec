import tensorflow as tf


def test_embedding(data):
    # color_data = {'color': [['R', 'G', 'B', 'A']],'age':[1]} 

    color_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'hist_i', ['1', '2', '3','4','5'], dtype=tf.string, default_value=-1
    )

    age_column = tf.feature_column.numeric_column('age')
    color_embeding = tf.feature_column.embedding_column(color_column, 8)
    net = tf.feature_column.input_layer(data, [age_column,color_embeding])

    # embedding = tf.reduce_mean(color_embeding_dense_tensor,1)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print('embeding' + '_' * 40)
        print(session.run([net]))

import tensorflow as tf
import numpy as np
_column = ['hist_i','y']
_ValueD = [[''],[0]]

def load_data():
    ds = tf.data.TextLineDataset("./data/train.csv").skip(1)

    COLUMNS = ['hist_i',
           'y']
    FIELD_DEFAULTS = [[''],[0]]
    def _parse_line(line):
        # Decode the line into its fields
        fields = tf.decode_csv(line, FIELD_DEFAULTS)

        # Pack the result into a dictionary
        features = dict(zip(COLUMNS,fields))
        hist_i = features.get('hist_i',None)
        print(hist_i)
        if hist_i is not None:
            arr = tf.string_split([hist_i],delimiter=' ')
        features['hist_i']=arr.values
        # Separate the label from the features
        label = features.pop('y')

        return features, label

    ds = ds.map(_parse_line)

    iterator = ds.make_initializable_iterator()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(iterator.initializer)

        print(session.run(iterator.get_next()))


load_data()
