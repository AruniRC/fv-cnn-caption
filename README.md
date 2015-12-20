# Generating feature encodings using deep CNNs

## Base code description

We use this code to generate CNN features on Flickr8k images using Imagenet pre-trained CNNs from VGG.
Since Flickr8k images do not have class labels, it is not possible to directly fine-tune a network on them. 
We therefore use a Gaussian Mixture Model to adapt to the Flickr8k images in an unsupervised manner, and then 
use Fisher Vector encoding to get features that are, hopefully, more adapted to the Flickr8k dataset than the 
original Imagenet pre-trained network features. This is under the assumption that there is a sufficient shift 
in domain between Imagenet and Flickr. 

This code simply extracts the features at various settings (details below). The training of LSTMs to perform 
image captioning on the FV-CNN and regular CNN features is the next step (not in this repo at present).


The repository contains code using VLFEAT and MATCONVNET to:

+ Train CNNs models from scratch or fine-tune on datasets
+ Extract a vareity of CNNs features including:
	+ R-CNN : features from CNNs at various layers
	+ D-CNN : CNN filterbanks with Fisher Vector pooling
	+ B-CNN : bilinear CNN
+ Run experiments on variety of datasets.


## Getting started with the code


### Compiling vlfeat

	cd vlfeat
	make MEX=/exp/comm/matlab-R2014b/bin/mex

### Compiling matconvnet

Checkout a stable release of the matconvnet. On my linux machines I found the v1.0-beta9 release to be stable. You can do this by:

	git fetch --tags
	git checkout tags/v1.0-beta9

Edit the Makefile to reflect the paths of CUDA and MATLAB. For example on my linux machine I set the following. 

	ARCH ?= glnxa64
	MATLABROOT ?= /exp/comm/matlab 
	CUDAROOT ?= /usr/local/cuda-7.0

I have a NVIDIA K40 GPU so I compiled matconvnet with GPU support. You may have to install libjpeg-dev to enable fast JPEG read support. On an ubuntu machines this is easily done by

	sudo apt-get install libjpeg-dev
	
I compiled the code using the following flags:
	
	make ENABLE_GPU=y ENABLE_IMREADJPEG=y
	

## Acknowledgements
The base code is taken from the Bilinear CNN project by Tsung-Yu Lin, Aruni RoyChowdhury and Subhransu Maji at UMass Amherst.






	

