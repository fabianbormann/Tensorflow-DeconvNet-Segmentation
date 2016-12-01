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

See `img_to_records_birds.py` for example of how to create `TFRecord`, 
and how to read in DeconvNet.py. Note that main has now moved to Deconvnet.py

The line:

`segmentation = tf.reshape(segmentation,[224,224])`

results in:

InvalidArgumentError (see above for traceback): Input to reshape is a tensor with 75264 values, but the requested shape has 50176

but I think it is very close!

--
Contributions welcome!
