from DeconvNet import *

# Remove use_cpu=True if you have enough GPU memory
deconvNet = DeconvNet(use_cpu=True)
deconvNet.train()