import os
import random
import tensorflow as tf

class FCN8_Segmentation:
    def read_PASCAL_VOC_2012_dataset(self):
        #self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def predict(self, image):
        return tf.cast(tf.argmax(self.prediction, 1), tf.float32).eval(session=self.session, feed_dict={self.x: [image]})[0]

    def predict_random_from_MNIST(self):
        try: self.mnist
        except AttributeError: self.read_MNIST_dataset()
        image = self.mnist.test.images[random.randint(0, 9999)]
        return (image, self.predict(image))

    def build(self, train=False, checkpoint_dir='./checkpoints/', training_steps=1000):
        self.x = tf.placeholder("float", shape=[224, 224, 3])
        rgb = tf.reshape(self.x, [-1, 224, 224, 3])

        conv_1_1 = self.conv_layer(rgb, 'conv_1_1')
        conv_1_2 = self.conv_layer(conv_1_1, 'conv_1_2')

        pool_1 = self.pool_layer(conv_1_2)

        conv_2_1 = self.conv_layer(pool_1, 'conv_2_1')
        conv_2_2 = self.conv_layer(conv_2_1, 'conv_2_2')

        pool_2 = self.pool_layer(conv_2_2)
  
        conv_3_1 = self.conv_layer(pool_2, 'conv_3_1')
        conv_3_2 = self.conv_layer(conv_3_1, 'conv_3_2')
        conv_3_3 = self.conv_layer(conv_3_2, 'conv_3_3')

        pool_3 = self.pool_layer(conv_3_3)

        conv_4_1 = self.conv_layer(pool_3, 'conv_4_1')
        conv_4_2 = self.conv_layer(conv_4_1, 'conv_4_2')
        conv_4_3 = self.conv_layer(conv_4_2, 'conv_4_3')

        pool_4 = self.pool_layer(conv_4_3)

        conv_5_1 = self.conv_layer(pool_4, 'conv_5_1')
        conv_5_2 = self.conv_layer(conv_5_1, 'conv_5_2')
        conv_5_3 = self.conv_layer(conv_5_2, 'conv_5_3')

        pool_5 = self.pool_layer(conv_5_3)

        fc_6 = self.conv_layer(pool_5, 'fc_6')
        fc_7 = self.conv_layer(fc_6, 'fc_7')

        fc_6_deconv = self.deconv_layer(fc_7, 'fc6_deconv')

        unpool_5 = self.unpool_layer(fc_6_deconv)

        deconv_5_3 = self.deconv_layer(unpool_5, 'deconv_5_3')
        deconv_5_2 = self.deconv_layer(deconv_5_3, 'deconv_5_2')
        deconv_5_1 = self.deconv_layer(deconv_5_2, 'deconv_5_1')

        unpool_4 = self.unpool_layer(deconv_5_1)

        deconv_4_3 = self.deconv_layer(unpool_4, 'deconv_4_3')
        deconv_4_2 = self.deconv_layer(deconv_4_3, 'deconv_4_2')
        deconv_4_1 = self.deconv_layer(deconv_4_2, 'deconv_4_1')        

        unpool_3 = self.unpool_layer(deconv_4_1)

        deconv_3_3 = self.deconv_layer(unpool_3, 'deconv_3_3')
        deconv_3_2 = self.deconv_layer(deconv_3_3, 'deconv_3_2')
        deconv_3_1 = self.deconv_layer(deconv_3_2, 'deconv_3_1')  

        unpool_2 = self.unpool_layer(deconv_3_1)

        deconv_2_2 = self.deconv_layer(unpool_2, 'deconv_2_2')
        deconv_2_1 = self.deconv_layer(deconv_2_2, 'deconv_2_1')  

        unpool_1 = self.unpool_layer(deconv_2_1)

        deconv_1_2 = self.deconv_layer(unpool_1, 'deconv_1_2')
        deconv_1_1 = self.deconv_layer(deconv_1_2, 'deconv_1_1') 

        seg_score_voc = self.conv_layer(deconv_1_1, 'seg_score_voc')

        ground_truth = tf.placeholder("float", shape=[224, 224])

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(ground_truth * tf.log(seg_score_voc), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(ground_truth, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        self.session = tf.Session()

        if train:
            try: self.mnist
            except AttributeError: self.read_MNIST_dataset()
            
            self.session.run(tf.initialize_all_variables())
            for i in range(training_steps):
                batch = self.mnist.train.next_batch(50)
                self.train_step.run(session=self.session, feed_dict={self.x: batch[0], y_: batch[1]})
                if i%100 == 0:
                    train_accuracy = self.accuracy.eval(session=self.session, feed_dict={self.x: batch[0], y_: batch[1]})
                    print("step {}, training accuracy {}".format(i, train_accuracy))
                    saver.save(self.session, checkpoint_dir+'model', global_step=i)

            print("test accuracy {}".format(self.accuracy.eval(session=self.session, feed_dict={self.x: self.mnist.test.images, y_: self.mnist.test.labels})))
            print("Model saved in file: ", checkpoint_dir)
        else:
            self.session.run(tf.initialize_all_variables())

            if not os.path.exists(checkpoint_dir):
                raise IOError(checkpoint_dir+' does not exist.')
            else:
                path = tf.train.get_checkpoint_state(checkpoint_dir)
                if path is None:
                    raise IOError('No checkpoint to restore in '+checkpoint_dir)
                else:
                    saver.restore(self.session, path.model_checkpoint_path)

    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv_layer(self, x, name):
        if name == 'conv_1':
            W = self.weight_variable([5, 5, 1, 32])
            b = self.bias_variable([32])
        elif name == 'conv_2':
            W = self.weight_variable([5, 5, 32, 64])
            b = self.bias_variable([64])
        else:
            raise ValueError(name+' is not part of the model')
        return tf.nn.relu(self.conv2d(x, W) + b)

    def pool_layer(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def fc_layer(self, x, name):
        if name == 'fc_6':
            W = self.weight_variable([7, 7, 512, 4096])
            b = self.bias_variable([4096])
        elif name == 'fc_7':
            W = self.weight_variable([1, 1, 4096, 1000])
            b = self.bias_variable([1000])
        elif name == 'fc_8':
            W = self.weight_variable([1, 1, 4096, 1000])
            b = self.bias_variable([1000])
        else:
            raise ValueError(name+' is not part of the model')
        return tf.nn.relu(tf.matmul(x, W) + b)

	def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001
            # Apply convolution
            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias