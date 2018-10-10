Tensorflow implementation of [Learning Deconvolution Network for Semantic Segmentation](http://arxiv.org/pdf/1505.04366v1.pdf).
## Install Instructions

1. Works with tensorflow 1.11.0 and uses the Keras API so use pip to install tensorflow-gpu in the latest version

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
>>> deconvNet.train(epochs=20, steps_per_epoch=500, batch_size=64)
>>> deconvNet.save()
>>> prediction = deconvNet.predict(any_image)
```

Contributions welcome!
