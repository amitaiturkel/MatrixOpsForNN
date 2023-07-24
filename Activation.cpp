#include "Activation.h" // Include the header file for the activation namespace

namespace activation
{
    // ReLU activation function implementation
    Matrix relu(const Matrix& matrix_x)
    {
      int cols = matrix_x.get_cols(); // Get the number of columns in the input matrix
      int rows = matrix_x.get_rows(); // Get the number of rows in the input matrix

      // Check if the input matrix has valid dimensions
      if (rows <= 0 || cols <= 0)
      {
        throw std::length_error("The matrix size is not valid.");
      }

      // Create a matrix to store the result of the ReLU activation
      matrix_dims result_dim = {rows, cols};
      Matrix result(result_dim);

      // Apply ReLU activation element-wise to the input matrix
      for (int row = 0; row < rows; row++)
      {
        for (int col = 0; col < cols; col++)
        {
          if (matrix_x(row, col) >= 0)
          {
            result(row, col) = matrix_x(row, col); // ReLU(x) = x if x >= 0
          }
          else
          {
            result(row, col) = 0.0F; // ReLU(x) = 0 if x < 0
          }
        }
      }

      return result; // Return the result matrix after applying ReLU activation
    }

    // Softmax activation function implementation
    Matrix softmax(const Matrix& matrix_x)
    {
      float exp_sum = 0; // Variable to store the sum of exponentials
      int cols = matrix_x.get_cols(); // Get the number of columns in the input matrix
      int rows = matrix_x.get_rows(); // Get the number of rows in the input matrix

      // Check if the input matrix has valid dimensions
      if (rows <= 0 || cols <= 0)
      {
        throw std::length_error("The matrix size is not valid.");
      }

      // Create a matrix to store the result of the softmax activation
      matrix_dims result_dim = {rows, cols};
      Matrix result(result_dim);

      // Apply softmax activation element-wise to the input matrix
      for (int row = 0; row < rows; row++)
      {
        for (int col = 0; col < cols; col++)
        {
          result(row, col) = std::exp(matrix_x(row, col)); // Compute the exponential of each element
          exp_sum += std::exp(matrix_x(row, col)); // Accumulate the sum of exponentials
        }
      }

      // Normalize the result by dividing each element by the sum of exponentials
      result = result * (1 / exp_sum);

      return result; // Return the result matrix after applying softmax activation
    }
}
