#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>
#include <string>
#include <fstream> // Required for std::ifstream
#include <iostream> // Required for std::cerr

// Template function to read binary data from a file
template <typename T>
void readBinaryFile(const std::string &filePath, std::vector<T> &data)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(T);
    file.seekg(0, std::ios::beg);
    data.resize(size);
    file.read(reinterpret_cast<char *>(data.data()), size * sizeof(T));
    file.close();
}

// Declaration of other functions
void convolve(const std::vector<float> &input, const std::vector<float> &kernel,
              const std::vector<float> &bias, std::vector<float> &output,
              int inputWidth, int inputHeight, int inputChannels, int outputChannels,
              int kernelSize, int stride, const std::string &padding);

void applyReLU(std::vector<float> &data);

#endif
