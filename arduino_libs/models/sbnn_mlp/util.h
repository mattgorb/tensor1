#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include "tensor1.h"



void t1_assert(int condition, const char* message);

//Call like this:   destroy_tensor(tensor2);
void destroy(void* x) ;
//Call like this: destroy_tensor_ptr(&tensor2);
void destroy_ptr(void** x) ;

void null_check(void* x) ;
void null_check_ptr(void** x);


float* test_image_mnist();
TensorFloat1D* tensor_test_image_mnist();

TensorInt161D* get_test_image_mnist_rounded(int image_num);

int16_t round_int8_t(double value);
#endif 