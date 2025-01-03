#ifndef POOLING_H
#define POOLING_H

#include <vector>

void maxPooling(const std::vector<float> &input, std::vector<float> &output,
                int inputWidth, int inputHeight, int channels, int poolSize, int stride);

#endif