# -*- coding: utf-8 -*-

__author__ = "shengyaqi"

import numpy as np
import tensorflow as tf
import ad_hoc_functions
import argparse
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser()
parser.add_argument("--dataDir",type=str,default='./data/1102/',help=u"directory to load data")
parser.add_argument('--walks', type=str, default='./data/1102/random_walk.npz' ,help=u"numpy serialized random walks. Use codes/pre_cumpute_walks.py to generate this file")
parser.add_argument('--log', type=str, default="./log/", help=u"directory to save tensorflow logs")
parser.add_argument('--save', type=str, default='./result/1102/', help=u"directory to save final embeddigs")
parser.add_argument('--embed_size',type=int,default=80,help=u"embeding size")
parser.add_argument('--context_window',type=int,default=5,help=u"context size")
parser.add_argument('--iter',type=int,default=3,help=u"iter")
parser.add_argument('--learning_rate',type=float,default=0.001,help=u"learning_rate")
args = parser.parse_args()

print("loading adjacent matrix")
tagSize  = 20
import json
dataDir = args.dataDir
## load itemIdxde
Id2IdxFile= open(dataDir+'itemId2Idx_map','r',encoding='utf-8')
idx2IdFile=open(dataDir+'itemIdx2Id_map','r',encoding='utf-8')
itemMetaFile =  open(dataDir+'item_meta_json','r',encoding='utf-8')
itemIdx = json.load(Id2IdxFile)
idxItem = json.load(idx2IdFile)
itemMeta = json.load(itemMetaFile)
## load tagIdx
tagMeta,tagIdx,tagIdxId = ad_hoc_functions.loadTagMeta(dataDir+"tag_idx")

## load item_meta_meta
item_tag_meta = ad_hoc_functions.loadItemTagMeta(dataDir+"item_tag_meta",itemIdx,tagIdx,tagSize)

### gen itemMeta.tsv

ad_hoc_functions.saveMeta(itemMeta,idxItem,tagMeta,tagIdxId,item_tag_meta,args.save+"item_meta.tsv",args.save+"tag_meta.tsv",dataDir+"item_tag_map")

print("loading pre-computed random walks")
random_walk_files=args.walks
np_random_walks=np.load(random_walk_files)['arr_0']
np.random.shuffle(np_random_walks)
num_tags = len(tagIdx)
num_nodes=len(itemIdx)
context_size= args.context_window
embedding_size = args.embed_size # Dimension of the embedding vector.
num_sampled = 64 # Number of negative examples to sample.
batch_size = None
loop = args.iter
num_random_walks=np_random_walks.shape[0]

print("num_random_walks:",num_random_walks)


print("defining compuotational graphs")
tf.logging.set_verbosity(tf.logging.INFO)


#Computational Graph Definition
# tag_node

graph = tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Parameters to learn
    # id embedding
    id_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
    # tag_embeddig
    tag_embeddings = tf.Variable(tf.random_uniform([num_tags, embedding_size], -1.0, 1.0))
    a_weights =  tf.Variable(tf.truncated_normal([num_nodes, tagSize+1],stddev=1.0 / math.sqrt(8)))
    softmax_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([num_nodes]))

    # Input data and re-orgenize size.
    with tf.name_scope("train_label") as scope:
        #context nodes to each input node in the batch (e.g [[1],[4],[5]] where batch_size = 3,context_size=3)
        train_context_node = tf.placeholder(tf.int32, shape=[batch_size,1],name="train_label")

    with tf.name_scope("train_input") as scope:
        #batch input node to the network(e.g [2,1,3] where batch_size = 3)
        train_input_node = tf.placeholder(tf.int32, shape=[batch_size,1],name="train_input")
        train_input_tag_node= tf.placeholder(tf.int32, shape=[batch_size,tagSize],name="train_input_tag")

    # Model.
    with tf.name_scope("hidden") as scope:
        # Look up embeddings for words.
        id_embed = tf.nn.embedding_lookup(id_embeddings, train_input_node,name="id_lookup")
        tag_embed = tf.nn.embedding_lookup(tag_embeddings, train_input_tag_node,name='tag_lookup')

    with tf.name_scope("loss") as scope:
        loss_node2vec = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights,softmax_biases,train_context_node,item_embed,num_sampled, num_nodes))
        loss_node2vec_summary = tf.summary.scalar("loss_node2vec", loss_node2vec)

    # Optimizer.
    #该函数返回以下结果
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    ##例： 以0.96为基数，每100000 步进行一次学习率的衰退
    # starter_learning_rate = args.learning_rate
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                        10000, 0.96, staircase=True)
    #    Passing global_step to minimize() will increment it at each step.
    update_loss = tf.train.AdamOptimizer(args.learning_rate).minimize(loss_node2vec,global_step=global_step)

    merged = tf.summary.merge_all()
    # Launch the graph

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=20)

import random

config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
num_skips = 2
average_loss = 0
with tf.Session(config=config,graph=graph) as sess:
    log_dir=args.log# tensorboard --logdir=./log1
    output_sir = args.save
    init.run()
    writer = tf.summary.FileWriter(log_dir+"graph", sess.graph)
    print("start run")
    for i in range(loop*num_random_walks):
        a_random_walk=np_random_walks[i%num_random_walks]
        input_batch,input_tag,labels= ad_hoc_functions.generate_batch(a_random_walk,context_size,item_tag_meta,num_skips,tagSize)
        try:
            if input_batch.shape[0] ==0  or input_tag.shape[0]==0 or labels.shape[0]==0:
                continue
        except:
            continue
            # print(a_random_walk)
        input_batch = np.reshape(input_batch,[len(input_batch),1])
        input_a_weight = input_batch.copy()
        # print(input_batch.shape,input_tag.shape,labels.shape,input_a_weight.shape)

        feed_dict={train_input_node:input_batch,train_input_tag_node:input_tag,train_context_node:labels,train_input_a_node:input_a_weight}
        _,loss_value,summary_str,id_vec,tag_vec,a_weight=sess.run([update_loss,loss_node2vec,merged,id_embeddings,tag_embeddings,a_weights], feed_dict)
        writer.add_summary(summary_str,i)
        # print('loss value echo batch', i, ': ', loss_value)
        average_loss +=loss_value
        if i % 2000 == 0:
            if i > 0:
                average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', i, ': ', average_loss)
            average_loss = 0
        if i == loop*num_random_walks-1:
            with open(output_sir+"id_embedding.tsv","w") as f:
                for vec in id_vec:
                    f.write('\t'.join(map(str, vec)))
                    f.write('\n')
            f.close()
            with open(output_sir+"tag_embedding.tsv","w") as f:
                for vec in tag_vec:
                    f.write('\t'.join(map(str, vec)))
                    f.write('\n')
            f.close()
            with open(output_sir + "a_weight.tsv",'w') as f:
                for vec in a_weight:
                    f.write('\t'.join(map(str,vec)))
                    f.write('\n')
            f.close()
    model_path=log_dir+"model.ckpt"
    save_path = saver.save(sess, model_path,global_step)
    print("Model saved in file: %s" % save_path)
