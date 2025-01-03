#include "convolution.h"
#include "dense.h"
#include "pooling.h"
#include <iostream>
#include <vector>
#include <string>

int main()
{
    // File paths for the first convolution layer
    std::string inputPath = "D:/MCW/Assignment-2/resized_image_binary.bin";
    std::string kernelPath1 = "D:/MCW/Assignment-1/data/weights/conv2d_kernel.bin";
    std::string biasPath1 = "D:/MCW/Assignment-1/data/weights/conv2d_bias.bin";

    // File paths for the second convolution layer
    std::string kernelPath2 = "D:/MCW/Assignment-1/data/weights/conv2d_1_kernel.bin";
    std::string biasPath2 = "D:/MCW/Assignment-1/data/weights/conv2d_1_bias.bin";

    // File paths for the third convolution layer
    std::string kernelPath3 = "D:/MCW/Assignment-1/data/weights/conv2d_2_kernel.bin";
    std::string biasPath3 = "D:/MCW/Assignment-1/data/weights/conv2d_2_bias.bin";

    // First layer dimensions
    const int inputWidth = 32;
    const int inputHeight = 32;
    const int inputChannels = 3;
    const int outputChannels1 = 64;
    const int kernelSize = 3;
    const int stride = 1;
    const std::string padding = "same";

    // Second layer dimensions
    const int outputChannels2 = 128;

    // Third layer dimensions
    const int outputChannels3 = 128;

    // Read input, kernel, and bias for the first layer
    std::vector<float> input, kernel1, bias1, convOutput1, maxPoolOutput1;
    readBinaryFile(inputPath, input);
    readBinaryFile(kernelPath1, kernel1);
    readBinaryFile(biasPath1, bias1);

    // Perform first convolution
    convolve(input, kernel1, bias1, convOutput1, inputWidth, inputHeight, inputChannels, outputChannels1, kernelSize, stride, padding);

    // Apply ReLU activation for the first layer
    applyReLU(convOutput1);

    // Perform MaxPooling for the first layer
    const int poolSize = 2;
    const int poolStride = 2;
    const int pooledWidth = inputWidth / poolStride;
    const int pooledHeight = inputHeight / poolStride;

    maxPooling(convOutput1, maxPoolOutput1, inputWidth, inputHeight, outputChannels1, poolSize, poolStride);

    // Read kernel and bias for the second layer
    std::vector<float> kernel2, bias2, convOutput2, maxPoolOutput2;
    readBinaryFile(kernelPath2, kernel2);
    readBinaryFile(biasPath2, bias2);

    // Perform second convolution
    convolve(maxPoolOutput1, kernel2, bias2, convOutput2, pooledWidth, pooledHeight, outputChannels1, outputChannels2, kernelSize, stride, padding);

    // Apply ReLU activation for the second layer
    applyReLU(convOutput2);

    // Perform MaxPooling for the second layer
    const int pooledWidth2 = pooledWidth / poolStride;
    const int pooledHeight2 = pooledHeight / poolStride;

    maxPooling(convOutput2, maxPoolOutput2, pooledWidth, pooledHeight, outputChannels2, poolSize, poolStride);

    // Read kernel and bias for the third layer
    std::vector<float> kernel3, bias3, convOutput3;
    readBinaryFile(kernelPath3, kernel3);
    readBinaryFile(biasPath3, bias3);

    // Perform third convolution
    convolve(maxPoolOutput2, kernel3, bias3, convOutput3, pooledWidth2, pooledHeight2, outputChannels2, outputChannels3, kernelSize, stride, padding);

    // Apply ReLU activation for the third layer
    applyReLU(convOutput3);
    // Dense layer dimensions
    const int denseInputSize = pooledWidth2 * pooledHeight2 * outputChannels3;
    const int denseOutputSize = 512;

    // File paths for Dense layer weights and biases
    std::string denseWeightsPath = "D:/MCW/Assignment-1/data/weights/dense_kernel.bin";
    std::string denseBiasesPath = "D:/MCW/Assignment-1/data/weights/dense_bias.bin";

    // Read weights and biases for Dense layer
    std::vector<float> denseWeights, denseBiases, denseOutput;
    readBinaryFile(denseWeightsPath, denseWeights);
    readBinaryFile(denseBiasesPath, denseBiases);

    // Perform Dense layer
    dense(convOutput3, denseWeights, denseBiases, denseOutput, denseInputSize, denseOutputSize);

    // Apply ReLU activation for Dense layer
    applyReLU(denseOutput);

    // For the second Dense layer
    std::string dense2WeightsPath = "D:/MCW/Assignment-1/data/weights/dense_1_kernel.bin";
    std::string dense2BiasesPath = "D:/MCW/Assignment-1/data/weights/dense_1_bias.bin";
    const int dense2InputSize = denseOutputSize; // Output of first dense layer is input to second
    const int dense2OutputSize = 10;

    std::vector<float> dense2Weights, dense2Biases, dense2Output;
    readBinaryFile(dense2WeightsPath, dense2Weights);
    readBinaryFile(dense2BiasesPath, dense2Biases);

    denseLayerWithSoftmax(denseOutput, dense2Weights, dense2Biases, dense2Output, dense2InputSize, dense2OutputSize);

    std::cout << "Output Shape after second Dense layer: (" << dense2OutputSize << ")" << std::endl;
    float ii = 0.0;
    float jj = 0.0;
    for (int i = 0; i < dense2OutputSize; ++i)
    {

        if (jj < dense2Output[i])
        {
            jj = dense2Output[i];
            ii = i;
        }
    }
    std::cout << "Predicted Class: " << ii << std::endl;
    switch (static_cast<int>(ii))
    {
    case 0:
        std::cout << "Airplane" << std::endl;
        break;
    case 1:
        std::cout << "Automobile" << std::endl;
        break;
    case 2:
        std::cout << "Bird" << std::endl;
        break;
    case 3:
        std::cout << "Cat" << std::endl;
        break;
    case 4:
        std::cout << "Deer" << std::endl;
        break;
    case 5:
        std::cout << "Dog" << std::endl;
        break;
    case 6:
        std::cout << "Frog" << std::endl;
        break;
    case 7:
        std::cout << "Horse" << std::endl;
        break;
    case 8:
        std::cout << "Ship" << std::endl;
        break;
    case 9:
        std::cout << "Truck" << std::endl;
        break;
    default:
        std::cout << "Unknown Class" << std::endl;
        break;
    }

    // 0: Airplane
    // 1: Automobile
    // 2: Bird
    // 3: Cat
    // 4: Deer
    // 5: Dog
    // 6: Frog
    // 7: Horse
    // 8: Ship
    // 9: Truck

    return 0;
}
// #include "convolution.h"
// #include "dense.h"
// #include "pooling.h"
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <nlohmann/json.hpp> // Include the nlohmann JSON library

// using json = nlohmann::json;

// int main()
// {
//     // File path for the model configuration JSON
//     std::string jsonPath = "D:/MCW/Assignment-2/model_config.json";

//     // Read and parse the JSON file
//     std::ifstream jsonFile(jsonPath);
//     if (!jsonFile.is_open()) {
//         std::cerr << "Failed to open JSON file: " << jsonPath << std::endl;
//         return -1;
//     }

//     json modelConfig;
//     jsonFile >> modelConfig;
//     jsonFile.close();

//     // Iterate through each layer defined in the JSON
//     std::vector<float> input;
//     for (const auto& layer : modelConfig["layers"]) {
//         std::string layerType = layer["type"];
//         std::vector<float> output;

//         if (layerType == "Conv2D") {
//             std::vector<float> kernel, bias;
//             std::string kernelPath = layer["weights_file_paths"][0];
//             std::string biasPath = layer["weights_file_paths"][1];
//             std::cout << "1" << std::endl;
//             readBinaryFile(kernelPath, kernel);
//             readBinaryFile(biasPath, bias);

//             int inputWidth = 32; // Assume initial input dimensions; update if dynamic
//             int inputHeight = 32;
//             int inputChannels = 3;
//             int outputChannels = 64;
//             int kernelSize = 3;
//             int stride = 1;
//             std::string padding = "same";

//             if (input.empty()) {
//                 readBinaryFile(layer["input_file_path"], input);
//             }

//             convolve(input, kernel, bias, output, inputWidth, inputHeight, inputChannels, outputChannels, kernelSize, stride, padding);
//             input = output; // Set output as the input for the next layer
//         } else if (layerType == "Activation") {
//             std::string activationType = layer["attributes"]["activation"];

//             if (activationType == "relu") {
//                 applyReLU(input);
//             }
//         } else if (layerType == "MaxPooling2D") {
//             int poolSize = 2;
//             int poolStride = 2;
//             int inputWidth = 32; // Update dynamically if needed
//             int inputHeight = 32;
//             int inputChannels = 64;

//             maxPooling(input, output, inputWidth, inputHeight, inputChannels, poolSize, poolStride);
//             input = output;
//         } else if (layerType == "Dense") {

//             std::vector<float> weights, biases;
//             std::string weightsPath = layer["weights_file_paths"][0];
//             std::string biasesPath = layer["weights_file_paths"][1];

//             readBinaryFile(weightsPath, weights);
//             readBinaryFile(biasesPath, biases);

//             int denseInputSize = input.size();
//             int denseOutputSize = weights.size() / denseInputSize;

//             std::cout << denseInputSize << std::endl;

//             if (layer["attributes"]["activation"] == "relu") {
//                 dense(input, weights, biases, output, denseInputSize, denseOutputSize);
//                 applyReLU(output);

//             } else if (layer["attributes"]["activation"] == "softmax") {
//                 denseLayerWithSoftmax(input, weights, biases, output, denseInputSize, denseOutputSize);
//             }

//             input = output;
//         }

//      else if (layerType == "Flatten") {
//             // Flattening is implicit in the input structure for dense layers
//             continue;
//         } else {
//             std::cerr << "Unsupported layer type: " << layerType << std::endl;
//             return -1;
//         }
//     }

//     // Output the final result and most probable class
//     std::cout << "Final Output: " << std::endl;
//     float maxVal = -1.0;
//     int maxIndex = -1;

//     for (size_t i = 0; i < input.size(); ++i) {
//         std::cout << input[i] << " ";
//         if (input[i] > maxVal) {
//             maxVal = input[i];
//             maxIndex = i;
//         }
//     }

//     std::cout << std::endl << "Predicted Class: " << maxIndex << std::endl;
//     std::string classNames[] = {"Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"};
//     std::cout << "Class Name: " << classNames[maxIndex] << std::endl;

//     return 0;
// }

////////////////////////// 2nd Method //////////////////////////

// #include "convolution.h"
// #include "dense.h"
// #include "pooling.h"
// #include <iostream>
// #include <vector>
// #include <string>
// #include <fstream>  // For file output

// // Function to write layer output to a file
// void writeLayerOutput(const std::string& layerName, const std::vector<float>& output, std::ofstream& outFile)
// {
//     outFile << layerName << " Output: " << std::endl;
//     for (size_t i = 0; i < output.size(); ++i)
//     {
//         outFile << output[i] << " ";
//     }
//     outFile << std::endl << std::endl;
// }

// int main()
// {
//     // File paths for the first convolution layer
//     std::string inputPath = "D:/MCW/Assignment-2/resized_image_binary.bin";
//     std::string kernelPath1 = "D:/MCW/Assignment-1/data/weights/conv2d_kernel.bin";
//     std::string biasPath1 = "D:/MCW/Assignment-1/data/weights/conv2d_bias.bin";

//     // File paths for the second convolution layer
//     std::string kernelPath2 = "D:/MCW/Assignment-1/data/weights/conv2d_1_kernel.bin";
//     std::string biasPath2 = "D:/MCW/Assignment-1/data/weights/conv2d_1_bias.bin";

//     // File paths for the third convolution layer
//     std::string kernelPath3 = "D:/MCW/Assignment-1/data/weights/conv2d_2_kernel.bin";
//     std::string biasPath3 = "D:/MCW/Assignment-1/data/weights/conv2d_2_bias.bin";

//     // Dense layer weights and biases
//     std::string denseWeightsPath = "D:/MCW/Assignment-1/data/weights/dense_kernel.bin";
//     std::string denseBiasesPath = "D:/MCW/Assignment-1/data/weights/dense_bias.bin";
//     std::string dense2WeightsPath = "D:/MCW/Assignment-1/data/weights/dense_1_kernel.bin";
//     std::string dense2BiasesPath = "D:/MCW/Assignment-1/data/weights/dense_1_bias.bin";

//     // Open output file for writing layer outputs
//     std::ofstream outFile("D:/MCW/Assignment-2/Assignment_2_test/CPP_LAYER_WISE_OUTPUT.txt", std::ios::out);
//     if (!outFile)
//     {
//         std::cerr << "Error opening file!" << std::endl;
//         return -1;
//     }

//     // First layer dimensions
//     const int inputWidth = 32;
//     const int inputHeight = 32;
//     const int inputChannels = 3;
//     const int outputChannels1 = 64;
//     const int kernelSize = 3;
//     const int stride = 1;
//     const std::string padding = "same";

//     // Second layer dimensions
//     const int outputChannels2 = 128;

//     // Third layer dimensions
//     const int outputChannels3 = 128;

//     // Read input, kernel, and bias for the first layer
//     std::vector<float> input, kernel1, bias1, convOutput1, maxPoolOutput1;
//     readBinaryFile(inputPath, input);
//     readBinaryFile(kernelPath1, kernel1);
//     readBinaryFile(biasPath1, bias1);

//     // Perform first convolution
//     convolve(input, kernel1, bias1, convOutput1, inputWidth, inputHeight, inputChannels, outputChannels1, kernelSize, stride, padding);
//     writeLayerOutput("Conv Layer 1", convOutput1, outFile);  // Output of first convolution layer

//     // Apply ReLU activation for the first layer
//     applyReLU(convOutput1);
//     writeLayerOutput("ReLU Layer 1", convOutput1, outFile);  // Output after ReLU

//     // Perform MaxPooling for the first layer
//     const int poolSize = 2;
//     const int poolStride = 2;
//     const int pooledWidth = inputWidth / poolStride;
//     const int pooledHeight = inputHeight / poolStride;

//     maxPooling(convOutput1, maxPoolOutput1, inputWidth, inputHeight, outputChannels1, poolSize, poolStride);
//     writeLayerOutput("MaxPooling Layer 1", maxPoolOutput1, outFile);  // Output after max pooling

//     // Second layer operations
//     std::vector<float> kernel2, bias2, convOutput2, maxPoolOutput2;
//     readBinaryFile(kernelPath2, kernel2);
//     readBinaryFile(biasPath2, bias2);

//     // Perform second convolution
//     convolve(maxPoolOutput1, kernel2, bias2, convOutput2, pooledWidth, pooledHeight, outputChannels1, outputChannels2, kernelSize, stride, padding);
//     writeLayerOutput("Conv Layer 2", convOutput2, outFile);  // Output of second convolution layer

//     // Apply ReLU activation for the second layer
//     applyReLU(convOutput2);
//     writeLayerOutput("ReLU Layer 2", convOutput2, outFile);  // Output after ReLU

//     // Perform MaxPooling for the second layer
//     const int pooledWidth2 = pooledWidth / poolStride;
//     const int pooledHeight2 = pooledHeight / poolStride;

//     maxPooling(convOutput2, maxPoolOutput2, pooledWidth, pooledHeight, outputChannels2, poolSize, poolStride);
//     writeLayerOutput("MaxPooling Layer 2", maxPoolOutput2, outFile);  // Output after second max pooling

//     // Third layer operations
//     std::vector<float> kernel3, bias3, convOutput3;
//     readBinaryFile(kernelPath3, kernel3);
//     readBinaryFile(biasPath3, bias3);

//     // Perform third convolution
//     convolve(maxPoolOutput2, kernel3, bias3, convOutput3, pooledWidth2, pooledHeight2, outputChannels2, outputChannels3, kernelSize, stride, padding);
//     writeLayerOutput("Conv Layer 3", convOutput3, outFile);  // Output of third convolution layer

//     // Apply ReLU activation for the third layer
//     applyReLU(convOutput3);
//     writeLayerOutput("ReLU Layer 3", convOutput3, outFile);  // Output after ReLU

//     // Dense layer operations
//     const int denseInputSize = pooledWidth2 * pooledHeight2 * outputChannels3;
//     const int denseOutputSize = 512;

//     std::vector<float> denseWeights, denseBiases, denseOutput;
//     readBinaryFile(denseWeightsPath, denseWeights);
//     readBinaryFile(denseBiasesPath, denseBiases);

//     // Perform Dense layer
//     dense(convOutput3, denseWeights, denseBiases, denseOutput, denseInputSize, denseOutputSize);
//     writeLayerOutput("Dense Layer 1", denseOutput, outFile);  // Output after first dense layer

//     // Apply ReLU activation for Dense layer
//     applyReLU(denseOutput);
//     writeLayerOutput("ReLU Dense Layer 1", denseOutput, outFile);  // Output after ReLU on Dense layer

//     // For the second Dense layer
//     const int dense2InputSize = denseOutputSize;
//     const int dense2OutputSize = 10;

//     std::vector<float> dense2Weights, dense2Biases, dense2Output;
//     readBinaryFile(dense2WeightsPath, dense2Weights);
//     readBinaryFile(dense2BiasesPath, dense2Biases);

//     denseLayerWithSoftmax(denseOutput, dense2Weights, dense2Biases, dense2Output, dense2InputSize, dense2OutputSize);
//     writeLayerOutput("Dense Layer 2", dense2Output, outFile);  // Output after second dense layer

//     // Print predicted class
//     std::cout << "Output Shape after second Dense layer: (" << dense2OutputSize << ")" << std::endl;
//     float maxValue = 0.0;
//     int predictedClass = 0;
//     for (int i = 0; i < dense2OutputSize; ++i)
//     {
//         if (maxValue < dense2Output[i])
//         {
//             maxValue = dense2Output[i];
//             predictedClass = i;
//         }
//     }
//     std::cout << "Predicted Class: " << predictedClass << std::endl;
//     switch (predictedClass) {
//         case 0: std::cout << "Airplane" << std::endl; break;
//         case 1: std::cout << "Automobile" << std::endl; break;
//         case 2: std::cout << "Bird" << std::endl; break;
//         case 3: std::cout << "Cat" << std::endl; break;
//         case 4: std::cout << "Deer" << std::endl; break;
//         case 5: std::cout << "Dog" << std::endl; break;
//         case 6: std::cout << "Frog" << std::endl; break;
//         case 7: std::cout << "Horse" << std::endl; break;
//         case 8: std::cout << "Ship" << std::endl; break;
//         case 9: std::cout << "Truck" << std::endl; break;
//         default: std::cout << "Unknown Class" << std::endl; break;
//     }

//     // Close output file
//     outFile.close();

//     return 0;
// }
