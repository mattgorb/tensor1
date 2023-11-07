#ifndef SBNN_FP_H
#define SBNN_FP_H


#include <stdio.h>
#include "tensor1.h"


//SBLinear_FP init_linear(int rows, int cols);
TensorFloat1D* sbnn_fp_forward(TensorFloat1D** input);
TensorFloat1D* linear1(TensorFloat1D** input);
TensorFloat1D* linear2(TensorFloat1D** input);
#endif 