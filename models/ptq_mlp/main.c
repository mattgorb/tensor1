#include <stdio.h>
#include "tensor1.h"
#include "util.h"
#include "ptq.h"

int main() {
    TensorInt161D* test_img=get_test_image_mnist_rounded(4);
    

    TensorInt161D* output=ptq_forward(&test_img);
    /*printf("Output: ");
    for(int i=0;i<10;i++){
        printf("%i, ",output->data[i]);
    }
    printf("\n");*/

    return 0;
}
