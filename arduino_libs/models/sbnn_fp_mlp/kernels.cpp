#include "kernels.h"
#include "util.h"

TensorFloat1D* fp_linear(struct QInt8Linear* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu) {
    if (layer->weight.data == NULL || input == NULL || output == NULL) {
        // Error handling
        printf("Invalid input or output pointers.\n");
        exit(EXIT_FAILURE);
    }

    if (layer->weight.cols != (*input)->shape || output->shape != layer->weight.rows) {
        printf("Incompatible dimensions: weight[%d][%d], input[%u], output[%u]\n", layer->weight.rows, layer->weight.cols, (*input)->shape, output->shape);
        exit(EXIT_FAILURE);
    }
    // Perform matrix-vector multiplication
    for (int i = 0; i < layer->weight.rows; i++) {
        output->data[i] = 0.0f;
        for (int j = 0; j < layer->weight.cols; j++) {
            output->data[i] +=  layer->weight.data[i][j] * (*input)->data[j];
        }
        if(fuse_relu==true){
            if(output->data[i]<0){
                output->data[i]=0;
            }
        }
    }

    return output;
}





TensorInt161D* qint8_linear(struct QInt8Linear* layer, TensorInt161D** input, TensorInt161D* output, bool fuse_relu) {
    if (layer->weight.data == NULL || input == NULL || output == NULL) {
        // Error handling
        printf("Invalid input or output pointers.\n");
        exit(EXIT_FAILURE);
    }

    if (layer->weight.cols != (*input)->shape || output->shape != layer->weight.rows) {
        printf("Incompatible dimensions: weight[%d][%d], input[%u], output[%u]\n", layer->weight.rows, layer->weight.cols, (*input)->shape, output->shape);
        exit(EXIT_FAILURE);
    }
    // Perform matrix-vector multiplication
    for (int i = 0; i < layer->weight.rows; i++) {
        output->data[i] = 0;
        for (int j = 0; j < layer->weight.cols; j++) {
            output->data[i] +=  layer->weight.data[i][j] * (*input)->data[j];
            
        }
        
        //quantization operations
        double activation=((output->data[i]-layer->zero_point) * layer->scale)+layer->bias.data[i];
        output->data[i]=round_int8_t(activation);
        
        if(fuse_relu==true){
            if(output->data[i]<0){  
                output->data[i]=0;
            }
        }

         
    }

    return output;
}
















TensorFloat1D* sb_fp_linear(struct SBLinear_FP* layer, TensorFloat1D** input, TensorFloat1D* output, bool fuse_relu) {
    if (layer->weight.data == NULL || input == NULL || output == NULL) {
        // Error handling
        printf("Invalid input or output pointers.\n");
        exit(EXIT_FAILURE);
    }

    if (layer->weight.cols != (*input)->shape || output->shape != layer->weight.rows) {
        printf("Incompatible dimensions: weight[%d][%d], input[%u], output[%u]\n", layer->weight.rows, layer->weight.cols, (*input)->shape, output->shape);
        exit(EXIT_FAILURE);
    }
    // Perform matrix-vector multiplication
    for (int i = 0; i < layer->weight.rows; i++) {
        output->data[i] = 0.0f;
        for (int j = 0; j < layer->weight.cols; j++) {
            output->data[i] +=  layer->weight.data[i][j] * (*input)->data[j]* layer->mask.data[i][j] * layer->alpha;
        }
        
        if(fuse_relu==true){
            if(output->data[i]<0){
                output->data[i]=0;
            }
        }
    }

    return output;
}





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




