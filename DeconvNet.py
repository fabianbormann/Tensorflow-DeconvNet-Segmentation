import os
import random
import tensorflow as tf
import time
import wget
import tarfile
import numpy as np
import cv2


class DeconvNet:
    def __init__(self, use_cpu=False, checkpoint_dir='./checkpoints/'):
        self.maybe_download_and_extract()

        self.build(use_cpu=use_cpu)

        self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        config = tf.ConfigProto(allow_soft_placement = True)
        self.session = tf.Session(config = config)
        self.session.run(tf.initialize_all_variables())
        self.checkpoint_dir = checkpoint_dir

    def maybe_download_and_extract(self):
        """Download and unpack VOC data if data folder only contains the .gitignore file"""
        if os.listdir('data') == ['.gitignore']:
            filenames = ['VOC_OBJECT.tar.gz', 'VOC2012_SEG_AUG.tar.gz', 'stage_1_train_imgset.tar.gz', 'stage_2_train_imgset.tar.gz']
            url = 'http://cvlab.postech.ac.kr/research/deconvnet/data/'

            for filename in filenames:
                wget.download(url + filename, out=os.path.join('data', filename))

                tar = tarfile.open(os.path.join('data', filename))
                tar.extractall(path='data')
                tar.close()

                os.remove(os.path.join('data', filename))

    def restore_session():
        global_step = 0
        if not os.path.exists(self.checkpoint_dir):
            raise IOError(self.checkpoint_dir + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if path is None:
                raise IOError('No checkpoint to restore in ' + self.checkpoint_dir)
            else:
                self.saver.restore(self.session, path.model_checkpoint_path)
                global_step = int(path.model_checkpoint_path.split('-')[-1])

        return global_step


    def predict(self, image):
        restore_session()
        return self.prediction.eval(session=self.session, feed_dict={image: [image]})[0]


    def train(self, train_stage=1, training_steps=1000, restore_session=False, learning_rate=1e-6):
        if restore_session:
            step_start = restore_session()
        else:
            step_start = 0

        if train_stage == 1:
            trainset = open('data/stage_1_train_imgset/train.txt').readlines()
        else:
            trainset = open('data/stage_2_train_imgset/train.txt').readlines()

        for i in range(step_start, step_start+training_steps):
            # pick random line from file
            random_line = random.choice(trainset)
            image_file = random_line.split(' ')[0]
            ground_truth_file = random_line.split(' ')[1]
            image = np.float32(cv2.imread('data' + image_file))
            ground_truth = cv2.imread('data' + ground_truth_file[:-1], cv2.IMREAD_GRAYSCALE)
            # norm to 21 classes [0-20] (see paper)
            ground_truth = (ground_truth / 255) * 20
            print('run train step: '+str(i))
            start = time.time()
            self.train_step.run(session=self.session, feed_dict={self.x: [image], self.y: [ground_truth], self.rate: learning_rate})

            if i % 10000 == 0:
                print('step {} finished in {:.2f} s with loss of {:.6f}'.format(
                    i, time.time() - start, self.loss.eval(session=self.session, feed_dict={self.x: [image], self.y: [ground_truth]})))
                self.saver.save(self.session, self.checkpoint_dir+'model', global_step=i)
                print('Model {} saved'.format(i))

    def build(self, use_cpu=False):
        '''
        use_cpu allows you to test or train the network even with low GPU memory
        anyway: currently there is no tensorflow CPU support for unpooling respectively
        for the tf.nn.max_pool_with_argmax metod so that GPU support is needed for training
        and prediction
        '''

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.device(device):
            self.x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
            self.y = tf.placeholder(tf.int64, shape=(1, None, None))
            expected = tf.expand_dims(self.y, -1)
            self.rate = tf.placeholder(tf.float32, shape=[])

            conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, 'conv_1_1')
            conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2')

            pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

            conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
            conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')

            pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

            conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
            conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
            conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')

            pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

            conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
            conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
            conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')

            pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

            conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
            conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
            conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')

            pool_5, pool_5_argmax = self.pool_layer(conv_5_3)

            fc_6 = self.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
            fc_7 = self.conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')

            deconv_fc_6 = self.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

            unpool_5 = self.unpool_layer2x2(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))

            deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
            deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
            deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

            unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))

            deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
            deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
            deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

            unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))

            deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
            deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
            deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

            unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))

            deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
            deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

            unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))

            deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
            deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

            score_1 = self.deconv_layer(deconv_1_1, [1, 1, 21, 32], 21, 'score_1')

            logits = tf.reshape(score_1, (-1, 21))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(expected, [-1]), name='x_entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

            self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

            self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
            self.accuracy = tf.reduce_sum(tf.pow(self.prediction - expected, 2))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b)

    def pool_layer(self, x):
        '''
        see description of build method
        '''
        with tf.device('/gpu:0'):
            return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

        return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.pack(output_list)

    def unpool_layer2x2(self, x, raveled_argmax, out_shape):
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.pack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat(3, [t2, t1])
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def unpool_layer2x2_batch(self, x, argmax):
        '''
        Args:
            x: 4D tensor of shape [batch_size x height x width x channels]
            argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
            values chosen for each output.
        Return:
            4D output tensor of shape [batch_size x 2*height x 2*width x channels]
        '''
        x_shape = tf.shape(x)
        out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

        batch_size = out_shape[0]
        height = out_shape[1]
        width = out_shape[2]
        channels = out_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

        x1 = tf.transpose(x, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
