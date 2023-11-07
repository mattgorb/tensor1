# tensor1

This repository is a C implementation of three types quantized neural networks: 
- Sparse + Binary Neural Network
- Post-Training 8-bit Quantization
- Quantization Aware Training

It was written as part of my PhD dissertation, specifically for the purpose of deployment on microcontrollers.  Each model has been tested on an Arduino Nano 33.  The models are each simple MLPs (single hidden layer) with 256 hidden neurons and a RelU activation.  I fuse the ReLU activation into the linear kernel for each model.  

The sparse binary neural network implements a special kernel which 1) unpacks bit values from uint_8 data structures and 2) iterates simultaneously over the mask and weights.  


I hope to get more time in the future to build out this repo with convolutional and attention layers. 

To make a model run either of the following: 
make build-arduino MODEL=sbnn_fp
make build-model MODEL=sbnn_fp
