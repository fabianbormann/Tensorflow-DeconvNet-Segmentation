import os
import random
import tensorflow as tf
import time
import wget
import tarfile
import numpy as np
import argparse

class DeconvNet:
    def __init__(self, images, segmentations, lr, use_cpu=False, checkpoint_dir='./checkpoints/'):
        #self.maybe_download_and_extract()
        
        self.x = images
        self.y = segmentations
        self.build(use_cpu=use_cpu)

        self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        self.checkpoint_dir = checkpoint_dir
        self.rate=lr
        start=time.time()

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

    # ! Does not work for now ! But currently I'm working on it -> PR's welcome!
    def predict(self, image):
        restore_session()
        return self.prediction.eval(session=self.session, feed_dict={image: [image]})[0]

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
            # Don't need placeholders when prefetching TFRecords
            #self.x = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='x_data')
            #self.y = tf.placeholder(tf.int64, shape=(None, None, None), name='y_data')

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

            unpool_5 = self.unpool_layer2x2(deconv_fc_6, self.unravel_argmax(pool_5_argmax, tf.to_int64(tf.shape(conv_5_3))))

            deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
            deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
            deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

            unpool_4 = self.unpool_layer2x2(deconv_5_1, self.unravel_argmax(pool_4_argmax, tf.to_int64(tf.shape(conv_4_3))))

            deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
            deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
            deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

            unpool_3 = self.unpool_layer2x2(deconv_4_1, self.unravel_argmax(pool_3_argmax, tf.to_int64(tf.shape(conv_3_3))))

            deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
            deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
            deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

            unpool_2 = self.unpool_layer2x2(deconv_3_1, self.unravel_argmax(pool_2_argmax, tf.to_int64(tf.shape(conv_2_2))))

            deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
            deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

            unpool_1 = self.unpool_layer2x2(deconv_2_1, self.unravel_argmax(pool_1_argmax, tf.to_int64(tf.shape(conv_1_2))))

            deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
            deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

            score_1 = self.deconv_layer(deconv_1_1, [1, 1, 21, 32], 21, 'score_1')

            logits = tf.reshape(score_1, (-1, 21))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(tf.reshape(self.y, [-1]), tf.int64), name='x_entropy')
            loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
            loss = tf.Print(loss, [loss], "loss: ")

            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

            self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
            # commented out for now, was throwing errors, calculating loss is fine.
            #self.accuracy = tf.reduce_sum(tf.pow(self.prediction - self.y, 2))

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

    def unpool_layer2x2(self, x, argmax):
        x_shape = tf.shape(x)
        output = tf.zeros([x_shape[1] * 2, x_shape[2] * 2, x_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [(width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, height // 2, width // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.pack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat(3, [t2, t1])
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def unpool_layer2x2_batch(self, bottom, argmax):
        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = self.unravel_argmax(argmax, argmax_shape)

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

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

if __name__ == '__main__':

    def read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'mask_raw': tf.FixedLenFeature([], tf.string),
            }
        )

        '''
        image = tf.decode_raw(features['image_raw'], tf.float32)
        segmentation = tf.decode_raw(features['mask_raw'], tf.int64)
        '''
        # must be read back as uint8 here
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)
        image.set_shape([224*224*3])
        segmentation.set_shape([224*224*1])

        '''
        image = tf.cast(image, tf.float32)
        segmentation = tf.cast(segmentation, tf.int64)
        '''

        image = tf.reshape(image,[224,224,3])
        segmentation = tf.reshape(segmentation,[224,224])
        
        return image, segmentation

    def input_pipeline(filenames, batch_size, num_epochs):
        filename_queue = tf.train.string_input_producer(
            [filenames], num_epochs=num_epochs,shuffle=False)

        image, label = read_and_decode(filename_queue)

        '''
        images_batch, labels_batch = tf.train.batch(
            [image, label], 
            enqueue_many=False,
            batch_size=batch_size,
            allow_smaller_final_batch=True,
            )

        return images_batch, labels_batch
        '''
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=True,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return images_batch, labels_batch

    def cast_inputs_from_pipeline(img, seg):
        return tf.cast(img,tf.float32), tf.cast(seg,tf.int64)

    # begin
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_record', help="training tfrecord file", default="input_data_ciona_crop.tfrecords")
    parser.add_argument('--batch_size', help="batch size", type=int, default=32)
    parser.add_argument('--num_epochs', help="number of epochs.", type=int, default=50)
    parser.add_argument('--lr',help="learning rate",type=float, default=1e-6)
    args = parser.parse_args()

    trn_images_batch, trn_segmentations_batch = input_pipeline(
                                                    args.train_record,
                                                    args.batch_size,
                                                    args.num_epochs)
    '''
    trn_images_batch, trn_segmentations_batch = cast_inputs_from_pipeline(
                                                    trn_images_batch,
                                                    trn_segmentations_batch)
    '''


    deconvnet = DeconvNet(trn_images_batch, trn_segmentations_batch, args.lr, use_cpu=False)
    config = tf.ConfigProto(allow_soft_placement = True)

    init = tf.initialize_all_variables()
    init_locals = tf.initialize_local_variables()

    with tf.Session(config=config) as sess:

        sess.run([init, init_locals])
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(0,2):        
            try:
                while not coord.should_stop():
                    sess.run(deconvnet.train_step)
                    print sess.run(deconvnet.prediction)
            
            except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'
            finally:
                coord.request_stop()
                coord.join(threads)
