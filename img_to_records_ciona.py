import numpy as np
import os
import os.path
import tensorflow as tf
from tqdm import tqdm
import random
import time
import re
from pprint import pprint


image_path='/scratch/gallowaa/224-ground-truthed/images/'
segmentation_path='/scratch/gallowaa/224-ground-truthed/segmentations/'
try:
    os.stat(path)
except:
    pass

def read_filelist(image_path, segmentation_path):
    
    filename_queue_images = tf.train.string_input_producer( \
        tf.train.match_filenames_once(image_path+"*.jpg"))

    filename_queue_segmentations = tf.train.string_input_producer( \
        tf.train.match_filenames_once(segmentation_path+"*.jpg"))
    
    return filename_queue_images, filename_queue_segmentations

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

filename_queue_images, filename_queue_segmentations = read_filelist(image_path, segmentation_path)  # read file list from directory

reader = tf.WholeFileReader()
_, img_value = reader.read(filename_queue_images)
_, seg_value = reader.read(filename_queue_segmentations)

i_img = tf.image.decode_jpeg(img_value) #reads images into tensor
s_img = tf.image.decode_jpeg(seg_value) #reads images into tensor

img_shape = tf.constant([224,224])
i_img = tf.image.resize_images(i_img, img_shape)
s_img = tf.image.resize_images(s_img, img_shape)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    filename = os.path.join('input_data_ciona_crop.tfrecords')
    print('Writing', filename)
    sess.run(init)
    coord = tf.train.Coordinator()
    writer = tf.python_io.TFRecordWriter(filename)
    threads = tf.train.start_queue_runners(coord=coord)

    iseq_len=50
    # show progress bar
    for index in tqdm(range(iseq_len)):  #(11648)
        try:
            image = i_img.eval()
            mask  = s_img.eval()
            image_raw = image.tostring()
            mask_raw  = image.tostring()
            example = tf.train.Example(
                # Example contains a features proto object
                features=tf.train.Features(
                    # Features contains a map of string to Feature proto objects
                    feature={
                        # A feature contains one of either a int64_list, float_list, or bytes_list
                        'image_raw': _bytes_feature(image_raw),
                        'mask_raw': _bytes_feature(mask_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())
        except tf.errors.OutOfRangeError:
                print 'Done training -- epoch limit reached'

    coord.request_stop()
    coord.join(threads)