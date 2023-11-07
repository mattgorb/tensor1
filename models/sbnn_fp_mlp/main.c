#include <stdio.h>
#include "tensor1.h"
#include "util.h"
#include "sbnn_fp.h"

int main() {
    TensorFloat1D* test_img=tensor_test_image_mnist();

    TensorFloat1D* output=sbnn_fp_forward(&test_img);
    printf("Output:");
    for(int i=0;i<10;i++){
        printf("%f, ",output->data[i]);
    }
    return 0;
}
