#include <stdio.h>
#include "tensor1.h"
#include "util.h"
#include "qat.h"

int main() {
    TensorUInt81D* test_img=get_test_image_mnist_int(1);

    /**/
    TensorFloat1D* output=qat_forward(&test_img);
    /*printf("Output: ");
    for(int i=0;i<10;i++){
        printf("%i, ",output->data[i]);
    }
    printf("\n");*/
    

    return 0;
}
