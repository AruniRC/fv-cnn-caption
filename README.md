

# Improving deep CNN image encodings for caption generation


## Overview

We use this code to generate CNN features on Flickr8k images using Imagenet pre-trained CNNs from VGG.
Since Flickr8k images do not have class labels, it is not possible to directly fine-tune a network on them. 
We therefore use a Gaussian Mixture Model to adapt to the Flickr8k images in an unsupervised manner, and then 
use Fisher Vector encoding to get features that are, hopefully, more adapted to the Flickr8k dataset than the 
original Imagenet pre-trained network features. This is under the assumption that there is a sufficient shift 
in domain between Imagenet and Flickr. 


### CNN features as Local descriptors

![FV-CNN model](downloads/fv_cnn.png)

The features from an intermediate convolutional layer can be regarded as a _H x W x K_ tensor, as 
shown in the figure above. Here, _K_ = 512, denoting the number of channels or features for this feature map.
_H_ and _W_ in this example are both 27, as we use 448x448 images as input and extract features from the _relu-5_ 
layer of a VGG-16 model.


### Spatial information

Since many descriptions of images are closely related to spatial location (e.g. "the dog jumped *over* the bar"), 
including explicit positional information by appending (x,y) coordinates to features is done here.



## Code description

This code simply extracts the features at various settings (details below). The training of LSTMs to perform 
image captioning on the FV-CNN and regular CNN features is the next step (**not in this repo** at present).


The repository contains code using VLFEAT and MATCONVNET to:

+ Train CNNs models from scratch or fine-tune on datasets
+ Extract a vareity of CNNs features including:
	+ R-CNN : features from CNNs at various layers
	+ D-CNN : CNN filterbanks with Fisher Vector pooling




## Experiments

### Implementation

Our implementation was segmented into two parts, the
Fisher Vector Convolutional Neural Network (FV-CNN)
and the Long Short Term Memory(LSTM) network. We
used MatConvNet toolbox to generate the CNN (VGG-
16) and FV-CNN descriptors. 

We used the Python library
NeuralTalk to implement the Long Short Term Memory
Network. 

As described in previous sections,
we use the CNN outputs of the last hidden layer to 
estimate the Gaussian Mixture Model for **Fisher encodings**. 

For our spatially **augmented Fisher CNN**, we also augment the
spatial co-ordinates to the CNN output vector. 

The **LSTM Network** is single layer with a dimensionality of 256. 
A softmax activation is used to predict words at a position one
by one.


### Dataset

We chose the [Flickr8k](http:
//nlp.cs.illinois.edu/HockenmaierGroup/
Framing_Image_Description/KCCA.html) 
dataset of eight thousand images. The
images in this data set focus on people or animals (mainly
dogs) performing some action. The images were cho-
sen from six different Flickr groups (Kids in Action,
Dogs in Action, Outdoor Activities, Action Photography,
Flickr-Social) and tend not to contain any well-known
people or locations, but were manually selected to depict
a variety of scenes and situations. We divide the data
set into 6k training images, 1k validation images and 1k
testing images. Each image is annotated with five different
reference sentences. 
 


### Training

![training loss](downloads/Loss.png)

We use batch size of 100 sentences for the LSTM training.
Each model is trained for 50 epochs. The training loss
for each batch for LSTM training is shown in the figure above.
We calculate perplexity on the validation dataset for each
epoch. The trained model after a particular epoch is saved
if the validation perplexity for that epoch is lower than any
of the previous epochs. Finally, we use the model with
the lowest recorded validation perplexity to calculate the
BLEU scores on the test set.



## Results

 We evaluated original CNN (CNN),
FV-CNN (FV) and spatially-augmented FV-CNN (saFV)
on the Flickr8k data set. The evaluation metric used was the
BLEU (Bilingual Evaluation Understudy) scores, commonly used in sucn evaluations.


| Model         | BLEU-1        | BLEU-2         | BLEU-3        | BLEU-4        |
| ------------- |:-------------:| :-------------:|:-------------:|:------------- |
| CNN      	| 	56.3 	| 38.0 		 | 	24.6 	 | 	16.2	 |
| FV      	| 	55.2 	| 37.1 		 | 	24.1 	 | 	15.8	 |
| Fv+(x,y)     	| 	57.8 	| 39.0 		 | 	25.3 	 | 	16.4	 |



Some qualitative examples of our captions. 

![training loss](downloads/dog.png)
	

## Acknowledgements

This was a term project done by Abhyuday Jagannatha, Aruni RoyChowdhury, Huaizu Jiang 
and Weilong Hu at **UMass Amherst, Fall 2015**.


The base code for FV encoding is taken from the Bilinear CNN project by Tsung-Yu Lin, Aruni RoyChowdhury and Subhransu Maji at UMass Amherst.






	

