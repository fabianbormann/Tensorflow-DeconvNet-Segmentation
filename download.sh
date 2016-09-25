cd data
mkdir VOC2012

# download and extract data necessary for training
cd VOC2012

# download and extract stage 1 training data
# To understand those stages read the section 
# 6.1 Training Data Construction from http://arxiv.org/pdf/1505.04366v1.pdf

# OSX:
# curl http://cvlab.postech.ac.kr/research/deconvnet/data/VOC_OBJECT.tar.gz -o VOC_OBJECT.tar.gz
# Linux:
wget http://cvlab.postech.ac.kr/research/deconvnet/data/VOC_OBJECT.tar.gz
tar -zxvf VOC_OBJECT.tar.gz
rm -rf VOC_OBJECT.tar.gz

# download and extract stage 2 training data
# OSX:
# curl http://cvlab.postech.ac.kr/research/deconvnet/data/VOC2012_SEG_AUG.tar.gz -o VOC2012_SEG_AUG.tar.gz
# Linux:
wget http://cvlab.postech.ac.kr/research/deconvnet/data/VOC2012_SEG_AUG.tar.gz
tar -zxvf VOC2012_SEG_AUG.tar.gz
rm -rf VOC2012_SEG_AUG.tar.gz

cd ..

# download and extract imagesets necessary for training
cd imagesets

# download and extract stage 1 training data
# OSX: 
# curl http://cvlab.postech.ac.kr/research/deconvnet/data/stage_1_train_imgset.tar.gz -o stage_1_train_imgset.tar.gz
# Linux
wget http://cvlab.postech.ac.kr/research/deconvnet/data/stage_1_train_imgset.tar.gz
tar -zxvf stage_1_train_imgset.tar.gz
rm -rf stage_1_train_imgset.tar.gz

# download and extract stage 2 training data
# OSX: 
# curl http://cvlab.postech.ac.kr/research/deconvnet/data/stage_2_train_imgset.tar.gz -o stage_2_train_imgset.tar.gz
# Linux
wget http://cvlab.postech.ac.kr/research/deconvnet/data/stage_2_train_imgset.tar.gz
tar -zxvf stage_2_train_imgset.tar.gz
rm -rf stage_2_train_imgset.tar.gz

cd ..