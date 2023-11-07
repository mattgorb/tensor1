#include <stdio.h>
#include "tensor1.h"
#include "util.h"
#include "sbnn_256_weights.h"
#include "sbnn.h"

int main() {

    
    TensorFloat1D* test_img=tensor_test_image_mnist();
    TensorFloat1D* output=sbnn_forward(&test_img);

    /*printf("Output: ");
    for(int i=0;i<10;i++){
        printf("%i, ",output->data[i]);
    }
    printf("\n");*/


    return 0;
}
