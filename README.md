# tensor1

This repository is a C implementation of three types quantized neural networks: 
- Sparse + Binary Neural Network (1-bit)
- Post-Training 8-bit Quantization
- Quantization Aware Training (8-bit)

It was written as part of my PhD dissertation, specifically for the purpose of deployment on microcontrollers.  Each model has been tested on an Arduino Nano 33.  The models are each simple MLPs (single hidden layer) with 256 hidden neurons and a RelU activation.  I fuse the ReLU activation into the linear kernel for each model.  

To go step-by-step through the implementation, start in the python_quantization folder.  

The sparse binary neural network implements a special kernel which 1) unpacks bit values from uint_8 data structures and 2) iterates simultaneously over the mask and weights.  


I hope to get more time in the future to build out this repo with convolutional and attention layers. 

To make a model run either of the following: 
```
make build-arduino MODEL=sbnn_fp
make build-model MODEL=sbnn_fp
```


Sparse binary kernel (from https://github.com/mattgorb/tensor1/blob/main/src/kernels.c#L158)

```

TensorFloat1D* sb_linear(struct SBLinear* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu) {
    if (layer->weight.data == NULL || input == NULL || output == NULL) {
        // Error handling
        printf("Invalid input or output pointers.\n");
        exit(EXIT_FAILURE);
    }

    if (layer->weight.cols <= ((*input)->shape/32) || output->shape != layer->weight.rows) {
        printf("Incompatible dimensions: weight[%d][%d], input[%u], output[%u]\n", layer->weight.rows, layer->weight.cols, (*input)->shape, output->shape);
        exit(EXIT_FAILURE);
    }


    for (int i = 0; i < layer->weight.rows; i++) {
        output->data[i] = 0.0f;
        for (int j = 0; j < layer->weight.cols; j++) {
            int k;
            if(j==(layer->weight.cols-1) ){
                k=((*input)->shape -((layer->weight.cols-1)*8)-1);
                
            }else{
                k=7;
            }
            int l=0;
            for (; k >= 0; k--) {
                int8_t weight = (2*((layer->weight.data[i][j] >> k) & 1)-1); // Extract the i-th bit and convert from binary to +-1
                uint8_t mask = (layer->mask.data[i][j] >> k) & 1; // Extract the i-th bit
                output->data[i] += ((*input)->data[j*8+l]) * mask * weight * layer->alpha;
                l=l+1;
            }

        }
        if(fuse_relu==true){
            if(output->data[i]<0){
                output->data[i]=0;
            }
        }
    }
    return output;
}
```
