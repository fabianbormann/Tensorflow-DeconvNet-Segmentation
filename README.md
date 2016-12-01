#prefetch branch 

Tensorflow implementation (approach) of [Learning Deconvolution Network for Semantic Segmentation](http://arxiv.org/pdf/1505.04366v1.pdf). 
## Install Instructions

```zsh
git clone https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation.git
cd Tensorflow-DeconvNet-Segmentation
sudo pip3 install -r requirements.txt

python3
```

```python
Python 3.5.2+ (default, Sep 22 2016, 12:18:14) 
[GCC 6.2.0 20160927] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from DeconvNet import DeconvNet 
>>> deconvNet = DeconvNet() # will start collecting the VOC2012 data
```

### Status

See `img_to_records_birds.py` for example of how to create `TFRecord`, 
and how to read in DeconvNet.py. Note that main has now moved to Deconvnet.py

The implementation is nearly working, currently getting `Shapes (5,) and (6,) are not compatible` in 
`unpool_layer2x2_batch` (same result when using non-batch equivalent), specifically on line `t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])`.

I added a small sample TFRecord `input_data_ciona_crop.tfrecords` you can use to reproduce.

--
Contributions welcome!
