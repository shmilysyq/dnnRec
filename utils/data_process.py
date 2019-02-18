import tensorflow as tf 
def process_features(raw_features,params):
    for col in params:
        several_values_columns_to_array(raw_features,col,'#')
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #     print(session.run(raw_features))

def several_values_columns_to_array(raw_features, feature_name, sep):
    raw_features[feature_name] = tf.sparse_tensor_to_dense(
            tf.string_split(raw_features[feature_name],sep),
            default_value='')
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #     print(session.run([split_data]))

data={"seq":['a,b,c','b']}
param = ["seq"]

# several_values_columns_to_array(data,"seq",',')

# _CSV_COLUMNS = ['item_hist','cat1_session','cat2_session','cat3_session','cat4_session','click_last_i','click_last_cat1','click_last_cat2','click_last_cat3','click_last_cat4','label']

# _SEQ_COLUMNS = ['item_hist','cat1_session','cat2_session','cat3_session','cat4_session']

# _CSV_COLUMN_DEFAULTS = [[''],[''],[''],[''],[''],[''],[''],[''],[''],[''],[0]]


# several_values_columns_to_array(data,"seq",',')

_CSV_COLUMNS = ['item_hist','cat1_session','click_last_i','click_last_cat1','label']

_SEQ_COLUMNS = ['item_hist','cat1_session']

_CSV_COLUMN_DEFAULTS = [[''],[''],[''],[''],[0]]

_NUM_EXAMPLES = {
    'train': 1083722,
    'validation': 2001,
}

def input_fn_test(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
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

feature,label = input_fn_test('./data/train',10,False,3)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([feature,label]))