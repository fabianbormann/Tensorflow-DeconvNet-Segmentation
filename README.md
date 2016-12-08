Tensorflow implementation of [Learning Deconvolution Network for Semantic Segmentation](http://arxiv.org/pdf/1505.04366v1.pdf).
## Install Instructions

1. Get a [Tensorflow version](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#pip-installation) that fits to your system

2. Run the following commands in your terminal

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

### Improving training
`python write-tfrecords/img_to_records_pascal.py`

Will write entire PASCAL VOC2012 dataset as TFRecord. Takes about 4mins @ 100it/s.

Default behaviour:
- assumes default dataset location from DeconvNet.py
- writes TFRecord to `tfrecords` folder
- Uses [resize_image_with_crop_or_pad](https://www.tensorflow.org/versions/r0.12/api_docs/python/image.html#resize_image_with_crop_or_pad) to make all images and segmentations fixed size of 224x224
- run with `-h` to see help and change defaults, will need to change `decode_png` to use image format other than png.

--
Contributions welcome!
