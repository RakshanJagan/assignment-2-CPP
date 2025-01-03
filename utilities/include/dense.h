// utilities/include/dense.h
#ifndef DENSE_H
#define DENSE_H

#include <vector>

void dense(const std::vector<float> &input, const std::vector<float> &weights, const std::vector<float> &biases,
           std::vector<float> &output, int inputSize, int outputSize);

void denseLayerWithSoftmax(const std::vector<float> &input, const std::vector<float> &weights,
                           const std::vector<float> &biases, std::vector<float> &output,
                           int inputSize, int outputSize);

#endif