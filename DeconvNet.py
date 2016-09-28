import os
import random
import tensorflow as tf
import time
import wget
import tarfile
import numpy as np
import cv2


class DeconvNet:
    def __init__(self, checkpoint_dir='./checkpoints/'):
        self.maybe_download_and_extract()
        self.build()

        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

        self.session = tf.Session()
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

    def predict(self, image):
        if not os.path.exists(self.checkpoint_dir):
            raise IOError(self.checkpoint_dir + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if path is None:
                raise IOError('No checkpoint to restore in ' + self.checkpoint_dir)
            else:
                self.saver.restore(self.session, path.model_checkpoint_path)

        return self.prediction.eval(session=self.session, feed_dict={image: [image]})[0]

    def train(self, training_steps=1000, restore_session=False):
        # TODO: change train method
        for i in range(0, training_steps):
            start = time.time()
            index = random.randint(1, 10)
            image = np.float32(cv2.imread('data/{}.png'.format(str(index).zfill(3)), 0))
            expected = np.float32(cv2.imread('data/{}.png'.format(str(index).zfill(3)), 0))
            self.train_step.run(session=self.session, feed_dict={image: [image], ground_truth: [expected]})

            error = self.accuracy.eval(session=self.session, feed_dict={image: [image], ground_truth: [expected]})

            print('step {} with trainset {} finished in {:.2f}s with error of {:.2%} ({} total) and loss {:.6f}'.format(
                i, index, time.time() - start, (error/(expected.shape[0]*expected.shape[1])), 
                int(error), loss.eval(session=session, feed_dict={x: [image], y: [expected]})))

            if i % 10 == 0:
                image = np.float32(cv2.imread('data/001.png', 0))
                output = self.session.run(self.prediction, feed_dict={x: [image]})
                cv2.imwrite('cache/output{}.png'.format(str(i).zfill(5)), np.uint8(output[0] * 255))

            if i % 100 == 0:
                self.saver.save(self.session, self.checkpoint_dir+'model', global_step=i)
                print('Model {} saved'.format(i))

    def build(self):
        image = tf.placeholder(tf.float32, shape=[224, 224, 3])
        ground_truth = tf.placeholder(tf.int64, shape=[224, 224, 3])

        rgb = tf.reshape(image, [-1, 224, 224, 3])

        conv_1_1 = self.conv_layer(rgb, [3, 3, 3, 64], 64, 'conv_1_1')
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

        deconv_fc_6 = self.deconv_layer(fc_7, [7, 7, 512, 4096], 4096, 'fc6_deconv')

        unpool_5 = self.unpool_layer2x2(deconv_fc_6, pool_5_argmax)

        deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
        deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
        deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

        unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax)

        deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
        deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
        deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

        unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax)

        deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
        deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
        deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

        unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax)

        deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
        deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

        unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax)

        deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
        deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

        score_1 = self.deconv_layer(deconv_1_1, [1, 1, 21, 32], 21, 'score_1')

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(score_1, ground_truth, name='Cross_Entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

        self.prediction = tf.argmax(score_1, dimension=3)
        self.accuracy = tf.reduce_sum(tf.pow(tf.to_float(self.prediction) - ground_truth, 2))

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
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.pack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, W_shape[2]])

        return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

    # waiting for better performance with fulture version of tf.unravel_index
    # ! Currently this method does not work as expected - see tests/UnpoolLayerTest.ipynb !
    # If someone found a solution a PR would be great.
    # https://github.com/tensorflow/tensorflow/issues/2075
    def unravel_index(self, indices, shape):
        indices = tf.expand_dims(indices, 0)
        shape = tf.expand_dims(shape, 1)
        strides = tf.cumprod(shape, reverse=True)
        strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
        return (indices // strides_shifted) % strides

    #  ! Currently not working -> see tests/UnpoolLayerTest.ipynb !
    # There is no tf.nn.max_pool_with_argmax function for CPU only
    # I will try to get this unpool mehtod working, but PR's are really welcome!
    #
    # Also waiting for a nicer (C++ GPU) implementation
    # https://github.com/tensorflow/tensorflow/issues/2169
    def unpool_layer2x2(self, x, argmax_from_pool_layer):
        x_shape = tf.shape(x)
        output_shape = [x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3]]
        argmax_shape = argmax_from_pool_layer.get_shape().as_list()

        unraveled_pool_map = tf.zeros(output_shape, dtype=tf.float32)

        pool_map = self.unravel_index(argmax_from_pool_layer, output_shape)

        # Build unraveled pool map
        # Zeros initialized tensor same size as 2*width x 2*height of x and contains
        # ones at positions from argmax_from_pool_layer
        for feature_map in range(argmax_shape[3]):
            for h in range(argmax_shape[1]):
                for w in range(argmax_shape[2]):
                    unraveled_pool_map[pool_map[0, 0, h, w, feature_map], pool_map[1, 0, h, w, feature_map]]

        # Multiply 2x2 field of unraveled pool map with index of x
        # => Sets all positions of the unraveled pool map to corresponding x value
        for feature_map in range(x_shape[3]):
            for h in range(x_shape[1]):
                for w in range(x_shape[2]):
                    unraveled_pool_map[0, (h * 2):2, (w * 2):2, feature_map] *= x[0, h, w, feature_map]

        return argmax_from_pool_layer
