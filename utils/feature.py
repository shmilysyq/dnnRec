import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder


def get_one_hot_column_with_vocabulary_file(feature_name, vocabulary_path):
    vocabulary_size = len(open(vocabulary_path, 'r', encoding='utf8').readlines())
    feature_col = tf.feature_column.categorical_column_with_vocabulary_file(feature_name, vocabulary_path,
                                                                            vocabulary_size)
    # one_hot_feature_column_with_vocabulary_file = tf.feature_column.indicator_column(feature_col)
    return feature_col


def test_categorical_column_with_vocabulary_list():
    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}  # 4行样本
    builder = _LazyBuilder(color_data)

    color_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_column_tensor = color_column._get_sparse_tensors(builder)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
    
    color_column_identy = tf.feature_column.indicator_column(color_column)
    color_dense_tensor = tf.feature_column.input_layer(color_data, [color_column_identy])

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))


def test_embedding():
    tf.set_random_seed(1)
    color_data = {'color_list': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']],'last_color':[['G'],['A'],['B'],['A']]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'color_list', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_column_2 = tf.feature_column.categorical_column_with_vocabulary_list(
        'last_color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_column_tensor = color_column._get_sparse_tensors(builder)

    color_list_embeding = tf.feature_column.embedding_column(color_column, 4, combiner='mean')
    last_color_embeding = tf.feature_column.embedding_column(color_column_2, 4, combiner='mean')


    color_embeding_dense_tensor = tf.feature_column.input_layer(color_data, [color_list_embeding])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print('embeding' + '_' * 40)
        print(session.run([color_embeding_dense_tensor]))


def test_shared_embedding_column_with_hash_bucket():
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = tf.feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    color_column2 = tf.feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_embed = tf.feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(color_column_embed)
    color_dense_tensor = tf.feature_column.input_layer(color_data, color_column_embed[0])
    color_dense_tensor1 = tf.feature_column.input_layer(color_data, color_column_embed[1])
    color_dense_tensor2 = tf.feature_column.input_layer(color_data, color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor,color_dense_tensor1,color_dense_tensor2]))

test_shared_embedding_column_with_hash_bucket()


# test_embedding()

# test_categorical_column_with_vocabulary_list()

