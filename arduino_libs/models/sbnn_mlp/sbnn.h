#ifndef SBNN_FP_H
#define SBNN_FP_H


#include <stdio.h>
#include "tensor1.h"


TensorFloat1D* sbnn_forward(TensorFloat1D** input);
TensorFloat1D* linear1(TensorFloat1D** input);
TensorFloat1D* linear2(TensorFloat1D** input);
#endif 