# tensor1

This repository is a C implementation of three types quantized neural networks: 
- Sparse + Binary Neural Network
- Post-Training 8-bit Quantization
- Quantization Aware Training

It was written as part of my PhD dissertation, specifically for the purpose of deployment on microcontrollers.  Each model has been tested on an Arduino Nano 33.  

The sparse binary neural network implements a special kernel which 1) unpacks bit values from uint_8 data structures and 2) iterates simultaneously
