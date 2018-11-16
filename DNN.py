import os
import json
import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self,config):
        self.config = config
        # Summary Writer
        self.train_writer = tf.summary.FileWriter(str(config['model_dir']) + '/train')
        self.eval_writer = tf.summary.FileWriter(str(config['model_dir']) + '/eval')
        # Building network
        self.init_placeholders()
        self.build_model()
        self.init_optimizer()
    def init_placeholders(self):
        # item label
        self.y = tf.placeholder(tf.int32,[None])
        # user's history item id
        self.hist_i = tf.placeholder(tf.int32,[None,None])
         # [B] valid length of `hist_i`
        self.sl = tf.placeholder(tf.int32, [None,])

        # learning rate
        self.lr = tf.placeholder(tf.float64, [])

        # whether it's training or not
        self.is_training = tf.placeholder(tf.bool, [])


    def build_model(self):
        n_class = self.config['item_count']
        item_emb_w = tf.get_variable("item_emb",shape=[self.config["item_count"],self.config["itemID_embedding_size"]])
        
        item_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_i)
        ## 求平均
        item_emb = tf.reduce_mean(item_emb,1)

        ## hidden layers
        for units in self.config["hidden_units"]:
            net = tf.layers.dense(item_emb,units,activation=tf.nn.relu)
        
        ## output layer
        # Compute logits (1 per class).
        self.logits = tf.layers.dense(net, self.config['item_count'], activation=None)

        # Compute predictions.
        predicted_classes = tf.argmax(self.logits, 1)
        if self.config["mode"] == "predict":
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(self.logits),
                'logits': self.logits,
            }
             
        # Compute loss.
        self.weights = tf.get_variable("nce_weight",shape=[n_class, units])
        self.biases = tf.get_variable("nce_biase",shape=[n_class])
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.weights,
                     biases=self.biases,
                     labels=self.y,
                     inputs=net,
                     num_sampled=10,
                     num_classes=n_class))
    
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=self.y,
                                   predictions=predicted_classes,
                                   name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        self.train_summary = tf.summary.merge([
            tf.summary.histogram('embedding/item_emb', item_emb),
            tf.summary.scalar('Training Loss', self.loss),
        ])
    
    def init_optimizer(self):
        # Gradients and SGD update operation for training the model
        # trainable_params = tf.trainable_variables()
        if self.config['optimizer'] == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        elif self.config['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.config['optimizer'] == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op= tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=global_step)

        # # Compute gradients of loss w.r.t. all trainable variables
        # gradients = tf.gradients(self.loss, trainable_params)

        # # Clip gradients by a given maximum_gradient_norm
        # clip_gradients, _ = tf.clip_by_global_norm(
        #     gradients, self.config['max_gradient_norm'])

        # # Update the model
        # self.train_op = self.opt.apply_gradients(
        #     zip(clip_gradients, trainable_params), global_step=self.global_step)
    
    def train(self, sess, feature, l, add_summary=False):

        input_feed = {
            self.y: feature[1],
            self.hist_i: feature[0],
            self.lr: l,
            self.is_training: True,
            }

        output_feed = [self.loss, self.train_op]

        if add_summary:
            output_feed.append(self.train_summary)

        outputs = sess.run(output_feed, input_feed)

        if add_summary:
            self.train_writer.add_summary(
                outputs[2], global_step=self.global_step.eval())

        return outputs[0]

    # def eval(self, sess, feature):
    #     res1 = sess.run(self.logits, feed_dict={
    #         self.hist_i: feature[0],
    #         self.is_training: False,
    #         })
    #     res2 = sess.run(self.logits, feed_dict={
    #         self.hist_i: feature[0],
    #         self.is_training: False,
    #         })
    #     return np.mean(res1 - res2 > 0)

    # def test(self, sess, uij):
    #     res1, att_1, stt_1 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
    #         self.u: uij[0],
    #         self.i: uij[1],
    #         self.hist_i: uij[3],
    #         self.hist_t: uij[4],
    #         self.sl: uij[5],
    #         self.is_training: False,
    #         })
    #     res2, att_2, stt_2 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
    #         self.u: uij[0],
    #         self.i: uij[2],
    #         self.hist_i: uij[3],
    #         self.hist_t: uij[4],
    #         self.sl: uij[5],
    #         self.is_training: False,
    #         })
    #     return res1, res2, att_1, stt_1, att_2, stt_1


        
    def save(self, sess):
        checkpoint_path = os.path.join(self.config['model_dir'], 'dnn')
        saver = tf.train.Saver()
        save_path = saver.save(
            sess, save_path=checkpoint_path, global_step=self.global_step.eval())
        json.dump(self.config,
                open('%s-%d.json' % (checkpoint_path, self.global_step.eval()), 'w'),
                indent=2)
        print('model saved at %s' % save_path, flush=True)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path, flush=True)