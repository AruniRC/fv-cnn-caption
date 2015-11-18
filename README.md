# Detecting rain in radar imagery with deep CNNs

## Base code description

The repository contains code using VLFEAT and MATCONVNET to:

+ Train CNNs models from scratch or fine-tune on datasets
+ Extract a vareity of CNNs features including:
	+ R-CNN : features from CNNs at various layers
	+ D-CNN : CNN filterbanks with fisher vector pooling
	+ B-CNN : bilinear CNN
+ Run experiments on variety of datasets.


## Getting started with the code

### Checkout the deep-face repository to your local directory 
    
    git clone git@bitbucket.org:smaji/deep-face.git
	git submodule init
	git submodule update

The code uses three submodules: vlfeat, matconvnet and matlab-helpers. The first two need to be compiled seperately.

### Compiling vlfeat

	cd vlfeat
	make MEX=/exp/comm/matlab-R2014b/bin/mex

### Compiling matconvnet

Checkout a stable release of the matconvnet. On my linux machines I found the v1.0-beta6 release to be stable. You can do this by:

	git fetch --tags
	git checkout tags/v1.0-beta6

Edit the Makefile to reflect the paths of CUDA and MATLAB. For example on my linux machine I set the following. 

	ARCH ?= glnxa64
	MATLABROOT ?= /exp/comm/matlab 
	CUDAROOT ?= /usr/local/cuda-6.5

I have a NVIDIA K40 GPU so I compiled matconvnet with GPU support. You may have to install libjpeg-dev to enable fast JPEG read support. On an ubuntu machines this is easily done by

	sudo apt-get install libjpeg-dev
	
I compiled the code using the following flags:
	
	make ENABLE_GPU=y ENABLE_IMREADJPEG=y
	
	
## Radar experiments







	

