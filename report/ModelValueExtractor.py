import tensorflow as tf
import numpy as np
import json

# Load JSON configuration
config_file_path = "D:/MCW/Assignment-2/Assignment_2_test/configs/json/model_config.json"
with open(config_file_path, "r") as json_file:
    config = json.load(json_file)

# Access layers array from config
layers = config["layers"]

# Function to write the first channel output to a text file
def write_first_channel(layer_output, layer_name, output_file_path):
    with open(output_file_path, "a") as file:
        if len(layer_output.shape) > 3:  # If output has more than one channel (e.g., 4D tensor)
            file.write(f"{layer_name} - First channel output:\n")
            file.write(str(layer_output[0, :, :, 0]))  # First channel of the first sample in the batch
        else:
            file.write(f"{layer_name} - Output (no channels):\n")
            file.write(str(layer_output))
        file.write("\n\n")  # Add a newline between layers

# Function to load input data
def load_input(file_path, shape):
    data = np.fromfile(file_path, dtype=np.float32)
    if shape != "Unknown":
        data = data.reshape(shape)
    return data

# Function to save output data
def save_output(file_path, data):
    data.tofile(file_path)

# Set output file path
output_file_path = "D:/MCW/Assignment-2/Assignment_2_test/MODEL_LAYER_WISE_OUTPUT.txt"

# Clear the output file before writing
open(output_file_path, "w").close()

# Process each layer
layer_output = None
for layer in layers:
    input_file = layer["input_file_path"]
    output_file = layer["output_file_path"]
    layer_name = layer["layer_name"]

    # Load input for the current layer
    if input_file and layer_output is None:  # Load initial input
        layer_output = load_input(input_file, layer["attributes"]["input_shape"])

    if layer["type"] == "Conv2D":
        # Load weights and biases
        kernel_path, bias_path = layer["weights_file_paths"]
        kernel = np.fromfile(kernel_path, dtype=np.float32)
        bias = np.fromfile(bias_path, dtype=np.float32)

        kernel_shape = (
            layer["attributes"]["kernel_size"][0],
            layer["attributes"]["kernel_size"][1],
            layer["attributes"]["input_shape"][3],
            layer["attributes"]["output_shape"][3],
        )
        kernel = kernel.reshape(kernel_shape)

        # Perform convolution
        strides = [1] + layer["attributes"]["strides"] + [1]
        padding = layer["attributes"]["padding"].upper()

        conv_output = tf.nn.conv2d(layer_output, kernel, strides=strides, padding=padding)
        conv_output = tf.nn.bias_add(conv_output, bias)

        # Update layer output
        layer_output = conv_output.numpy()

    elif layer["type"] == "Activation":
        activation = layer["attributes"]["activation"]
        if activation == "relu":
            layer_output = tf.nn.relu(layer_output).numpy()
        elif activation == "softmax":
            layer_output = tf.nn.softmax(layer_output).numpy()

    elif layer["type"] == "MaxPooling2D":
        pool_size = [1, 2, 2, 1]  # Default pool size
        strides = [1] + layer["attributes"]["strides"] + [1]
        padding = layer["attributes"]["padding"].upper()

        layer_output = tf.nn.max_pool2d(layer_output, ksize=pool_size, strides=strides, padding=padding).numpy()

    elif layer["type"] == "Flatten":
        layer_output = layer_output.reshape((layer_output.shape[0], -1))

    elif layer["type"] == "Dense":
        # Load weights and biases
        weights_path, bias_path = layer["weights_file_paths"]
        weights = np.fromfile(weights_path, dtype=np.float32).reshape(
            (layer["attributes"]["input_shape"][1], layer["attributes"]["output_shape"][1])
        )
        biases = np.fromfile(bias_path, dtype=np.float32)

        # Perform dense computation
        layer_output = tf.matmul(layer_output, weights) + biases

        # Apply activation
        activation = layer["attributes"]["activation"]
        if activation == "relu":
            layer_output = tf.nn.relu(layer_output).numpy()
        elif activation == "softmax":
            layer_output = tf.nn.softmax(layer_output).numpy()

    # Write first channel output to the file
    write_first_channel(layer_output, layer_name, output_file_path)

    # Save output to file
    save_output(output_file, layer_output)
