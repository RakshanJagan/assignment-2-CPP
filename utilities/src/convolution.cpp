#include "convolution.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Function to read binary data from a file
//template <typename T>
// void readBinaryFile(const std::string &filePath, std::vector<T> &data)
// {
//     std::ifstream file(filePath, std::ios::binary);
//     if (!file.is_open())
//     {
//         std::cerr << "Error: Cannot open file " << filePath << std::endl;
//         exit(EXIT_FAILURE);
//     }
//     file.seekg(0, std::ios::end);
//     size_t size = file.tellg() / sizeof(T);
//     file.seekg(0, std::ios::beg);
//     data.resize(size);
//     file.read(reinterpret_cast<char *>(data.data()), size * sizeof(T));
//     file.close();
// }

// Perform convolution

// Perform convolution
void convolve(const std::vector<float> &input, const std::vector<float> &kernel,
              const std::vector<float> &bias, std::vector<float> &output,
              int inputWidth, int inputHeight, int inputChannels, int outputChannels,
              int kernelSize, int stride, const std::string &padding)
{
    // Padding calculation based on 'same' or 'valid'
    int pad = (padding == "same") ? kernelSize / 2 : 0;

    // Output dimensions, assuming padding doesn't change input width and height for 'same'
    int outputWidth = inputWidth;
    int outputHeight = inputHeight;

    // Resize output vector to match the required output size
    output.resize(outputWidth * outputHeight * outputChannels);

    for (int oc = 0; oc < outputChannels; ++oc)
    {
        for (int oh = 0; oh < outputHeight; ++oh)
        {
            for (int ow = 0; ow < outputWidth; ++ow)
            {
                float sum = bias[oc];  // Start with the bias value for the output channel
                for (int ic = 0; ic < inputChannels; ++ic)
                {
                    for (int kh = 0; kh < kernelSize; ++kh)
                    {
                        for (int kw = 0; kw < kernelSize; ++kw)
                        {
                            // Calculate the corresponding input pixel (taking padding into account)
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;

                            // Check if the input coordinates are valid (within the input image dimensions)
                            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                            {
                                // Calculate indices for input and kernel vectors
                                int inputIdx = (ic * inputHeight + ih) * inputWidth + iw;
                                int kernelIdx = ((oc * inputChannels + ic) * kernelSize + kh) * kernelSize + kw;

                                // Ensure indices are within valid range before accessing vectors
                                if (inputIdx >= 0 && inputIdx < input.size() &&
                                    kernelIdx >= 0 && kernelIdx < kernel.size())
                                {
                                    sum += input[inputIdx] * kernel[kernelIdx];
                                }
                            }
                        }
                    }
                }

                // Calculate output index
                int outputIdx = (oc * outputHeight + oh) * outputWidth + ow;
                // Ensure output index is valid before writing
                if (outputIdx >= 0 && outputIdx < output.size())
                {
                    output[outputIdx] = sum;
                }
            }
        }
    }
}

void applyReLU(std::vector<float> &data)
{
    for (float &value : data)
    {
        value = std::max(0.0f, value);
    }
}