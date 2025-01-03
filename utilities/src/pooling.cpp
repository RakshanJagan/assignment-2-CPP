#include "pooling.h"
#include <algorithm>
#include <limits>
// Perform MaxPooling
void maxPooling(const std::vector<float> &input, std::vector<float> &output,
                int inputWidth, int inputHeight, int channels, int poolSize, int stride)
{
    int outputWidth = inputWidth / stride;
    int outputHeight = inputHeight / stride;

    output.resize(outputWidth * outputHeight * channels);

    for (int c = 0; c < channels; ++c)
    {
        for (int oh = 0; oh < outputHeight; ++oh)
        {
            for (int ow = 0; ow < outputWidth; ++ow)
            {
                float maxValue = -std::numeric_limits<float>::infinity();
                for (int ph = 0; ph < poolSize; ++ph)
                {
                    for (int pw = 0; pw < poolSize; ++pw)
                    {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;
                        if (ih < inputHeight && iw < inputWidth)
                        {
                            int inputIdx = (c * inputHeight + ih) * inputWidth + iw;
                            maxValue = std::max(maxValue, input[inputIdx]);
                        }
                    }
                }
                int outputIdx = (c * outputHeight + oh) * outputWidth + ow;
                output[outputIdx] = maxValue;
            }
        }
    }
}