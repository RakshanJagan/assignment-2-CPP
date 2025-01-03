#include "dense.h"
#include <cmath>
#include <algorithm>



void dense(const std::vector<float> &input, const std::vector<float> &weights, const std::vector<float> &biases,
           std::vector<float> &output, int inputSize, int outputSize)
{
    output.resize(outputSize, 0.0f);
    for (int o = 0; o < outputSize; ++o)
    {
        for (int i = 0; i < inputSize; ++i)
        {
            output[o] += input[i] * weights[o * inputSize + i];
        }
        output[o] += biases[o];
    }
}

void denseLayerWithSoftmax(const std::vector<float> &input, const std::vector<float> &weights,
                           const std::vector<float> &biases, std::vector<float> &output,
                           int inputSize, int outputSize)
{
    // Initialize the output vector
    output.resize(outputSize, 0.0f);

    // Perform the dense operation (matrix multiplication and bias addition)
    for (int o = 0; o < outputSize; ++o)
    {
        for (int i = 0; i < inputSize; ++i)
        {
            output[o] += input[i] * weights[o * inputSize + i];
        }
        output[o] += biases[o];
    }

    // Apply softmax activation
    float maxVal = *std::max_element(output.begin(), output.end());
    float sumExp = 0.0f;

    for (float &value : output)
    {
        value = std::exp(value - maxVal); // Subtract maxVal for numerical stability
        sumExp += value;
    }

    for (float &value : output)
    {
        value /= sumExp;
    }
}
// #include "dense.h"
// #include <cmath>
// #include <algorithm>
// #include <stdexcept>
// #include <iostream>

// void dense(const std::vector<float> &input, const std::vector<float> &weights, const std::vector<float> &biases,
//            std::vector<float> &output, int inputSize, int outputSize)
// {
//     if (input.size() != inputSize) {
//         std::cerr << "Error: Input size mismatch. Expected: " << inputSize << ", Got: " << input.size() << std::endl;
//         throw std::invalid_argument("Input size does not match expected size.");
//     }
//     if (weights.size() != inputSize * outputSize) {
//         std::cerr << "Error: Weights size mismatch. Expected: " << (inputSize * outputSize) << ", Got: " << weights.size() << std::endl;
//         throw std::invalid_argument("Weights size does not match expected size.");
//     }
//     if (biases.size() != 512) {
//         std::cerr << "Error: Biases size mismatch. Expected: " << outputSize << ", Got: " << biases.size() << std::endl;
//         throw std::invalid_argument("Biases size does not match expected size.");
//     }

//     output.assign(outputSize, 0.0f);

//     for (int o = 0; o < 512; ++o) {
//         for (int i = 0; i < 8192; ++i) {
//             output[o] += input[i] * weights[o * inputSize + i];
//         }
//         output[o] += biases[o];
//     }
//     output.resize(outputSize, 0.0f);
// //     for (int o = 0; o < outputSize; ++o)
// //     {
// //         for (int i = 0; i < inputSize; ++i)
// //         {
// //             output[o] += input[i] * weights[o * inputSize + i];
// //         }
// //         output[o] += biases[o];
// //     }
// }


// void denseLayerWithSoftmax(const std::vector<float> &input, const std::vector<float> &weights,
//                            const std::vector<float> &biases, std::vector<float> &output,
//                            int inputSize, int outputSize)
// {
//     // Ensure input, weights, and biases have valid dimensions
//     if (input.size() != inputSize)
//         throw std::invalid_argument("Input size does not match expected size.");
//     if (weights.size() != inputSize * outputSize)
//         throw std::invalid_argument("Weights size does not match expected size.");
//     if (biases.size() != outputSize)
//         throw std::invalid_argument("Biases size does not match expected size.");

//     // Initialize the output vector
//     output.assign(outputSize, 0.0f);

//     // Perform dense computation
//     for (int o = 0; o < outputSize; ++o)
//     {
//         for (int i = 0; i < inputSize; ++i)
//         {
//             output[o] += input[i] * weights[o * inputSize + i];
//         }
//         output[o] += biases[o];
//     }

//     // Apply softmax activation
//     float maxVal = *std::max_element(output.begin(), output.end());
//     float sumExp = 0.0f;

//     for (float &value : output)
//     {
//         value = std::exp(value - maxVal); // Subtract maxVal for numerical stability
//         sumExp += value;
//     }

//     for (float &value : output)
//     {
//         value /= sumExp;
//     }
// }
