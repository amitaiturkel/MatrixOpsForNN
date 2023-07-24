# MatrixOpsForNN
"MatrixOpsForNN" is a C++ library for Neural Network implementations, streamlining matrix operations. It offers an efficient Matrix class with essential functionalities as building blocks for NN computations. Additionally, it includes Dense layers with customizable activation functions.
# Matrix and Neural Network Layers

This project contains C++ implementations of a Matrix class and neural network layers, including Dense layers with activation functions.

## Introduction

This project aims to provide a basic implementation of a Matrix class and neural network layers for educational purposes. The Matrix class enables operations on matrices, and the Dense class represents a dense (fully connected) layer in a neural network with customizable activation functions.

## Matrix Class

The Matrix class allows you to create, manipulate, and perform various operations on matrices. It includes functionalities such as matrix addition, multiplication, transpose, element-wise operations, and more. The class provides a straightforward interface for working with matrices.

## Dense Class

The Dense class represents a dense (fully connected) layer in a neural network. It takes weight and bias matrices as well as an activation function as input. The class supports applying the dense layer to input data and extracting weights, bias, and the activation function used.

## Activation Functions

The project includes two common activation functions implemented within the "activation" namespace:
- ReLU (Rectified Linear Unit): relu(x) = max(0, x)
- Softmax: softmax(x) = exp(x) / sum(exp(x))

## Usage

To use this project, include the necessary header files (Matrix.h, Dense.h, and Activation.h) in your C++ project. You can then create and manipulate matrices using the Matrix class, construct neural network layers with the Dense class, and apply activation functions from the activation namespace.

Examples of using the Matrix and Dense classes can be found in the code provided.

## Contributing

Oriyan Hassidim(the tests)
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

The Matrix and Neural Network Layers project is licensed under the MIT License.

## Contact

If you have any questions or inquiries, please feel free to contact [amitai.turkel] at [amitai.turkel@gmail.com].
