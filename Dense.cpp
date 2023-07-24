#include "Dense.h" // Include the header file for the Dense class

// Constructor for the Dense class
Dense::Dense(Matrix weights, Matrix bias, ActivationFunction func)
{
  // Check if the bias has the correct dimensions to match the weights
  if (bias.get_cols() == 1 && bias.get_rows() == weights.get_rows())
  {
    weight = weights; // Assign weights to the class member variable weight
    bi = bias; // Assign bias to the class member variable bi
    function = func; // Assign the activation function to the class member variable function
  }
  else
  {
    throw std::length_error("Invalid dimensions for the bias matrix.");
  }
}

// Get the weights matrix
Matrix Dense::get_weights() const
{
  return weight;
}

// Get the bias matrix
Matrix Dense::get_bias() const
{
  return bi;
}

// Get the activation function
ActivationFunction Dense::get_activation() const
{
  return function;
}

// Operator overload for the Dense class, which applies the Dense layer to the input matrix
Matrix Dense::operator()(Matrix input) const
{
  // Perform the dense layer operation: activation_function((weights * input) + bias)
  return function((weight * input) + bi);
}
