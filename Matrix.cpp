#include "Matrix.h" // Include the header file for the Matrix class

// Default constructor
Matrix::Matrix() : dimensions{1, 1}
{
  data = new float*[dimensions.rows];
  data[0] = new float[dimensions.cols];
  data[0][0] = 0.0F;
}

// Constructor with custom dimensions
Matrix::Matrix(matrix_dims dims) : dimensions { dims.rows, dims.cols }
{
  if (dims.cols <= 0 || dims.rows <= 0) {
    throw std::length_error("The matrix size is not valid.");
  }

  data = new float*[dimensions.rows];
  for (int i = 0; i < dimensions.rows; i++)
  {
    data[i] = new float[dimensions.cols];
    for (int j = 0; j < dimensions.cols; j++)
    {
      data[i][j] = 0.0F;
    }
  }
}

// Constructor with separate row and column arguments
Matrix::Matrix(int rows, int cols) : dimensions({rows, cols})
{
  if (rows <= 0 || cols <= 0) {
    throw std::length_error("The matrix size is not valid.");
  }

  data = new float*[rows];
  for (int i = 0; i < rows; i++)
  {
    data[i] = new float[cols];
    for (int j = 0; j < cols; j++)
    {
      data[i][j] = 0.0F;
    }
  }
}

// Copy constructor
Matrix::Matrix(const Matrix& other) noexcept : dimensions(other.dimensions)
{
  dimensions = other.dimensions;
  data = new float*[dimensions.rows];
  for (int i = 0; i < dimensions.rows; i++)
  {
    data[i] = new float[dimensions.cols];
    for (int j = 0; j < dimensions.cols; j++)
    {
      data[i][j] = other.data[i][j];
    }
  }
}

// Destructor, frees memory allocated for data
Matrix::~Matrix()
{
  for (int i = 0; i < dimensions.rows; i++)
  {
    delete[] data[i];
  }
  delete[] data;
}

// Get the number of rows in the matrix
int Matrix::get_rows() const
{
  return dimensions.rows;
}

// Get the number of columns in the matrix
int Matrix::get_cols() const
{
  return dimensions.cols;
}

// Transpose the matrix in place
Matrix& Matrix::transpose()
{
  matrix_dims trans_dim;
  trans_dim.rows = dimensions.cols;
  trans_dim.cols = dimensions.rows;
  Matrix trans(trans_dim);
  for (int row = 0; row < dimensions.rows; row++)
  {
    for (int col = 0; col < dimensions.cols; col++)
    {
      trans.data[col][row] = data[row][col];
    }
  }
  *this = trans;
  return *this;
}

// Calculate the sum of all elements in the matrix
float Matrix::sum() const
{
  float sum = 0.0F;
  for (int i = 0; i < dimensions.rows; i++)
  {
    for (int j = 0; j < dimensions.cols; j++)
    {
      sum += data[i][j];
    }
  }
  return sum;
}

// Vectorize the matrix (convert to a column vector)
Matrix& Matrix::vectorize()
{
  matrix_dims vecs;
  vecs.rows = dimensions.rows * dimensions.cols;
  vecs.cols = 1;
  Matrix vec(vecs);
  for (int row = 0; row < dimensions.rows; row++)
  {
    for (int col = 0; col < dimensions.cols; col++)
    {
      vec.data[col + (row * dimensions.cols)][0] = data[row][col];
    }
  }
  *this = vec;
  return *this;
}

// Print the matrix elements in a plain format
void Matrix::plain_print() const
{
  for (int row = 0; row < dimensions.rows; row++)
  {
    if (row != 0)
    {
      std::cout << std::endl;
    }

    for (int col = 0; col < dimensions.cols; col++)
    {
      std::cout << data[row][col] << " ";
    }
  }
  std::cout << std::endl;
}

// Matrix multiplication (dot product) with another matrix
Matrix Matrix::dot(const Matrix& other) const
{
  if (dimensions.cols != other.dimensions.rows ||
      dimensions.rows != other.dimensions.rows) {
    throw std::length_error("The matrix size is not valid.");
  }
  matrix_dims result_dim;
  result_dim.rows = dimensions.rows;
  result_dim.cols = dimensions.cols;

  Matrix result(result_dim);
  for (int row = 0; row < dimensions.rows; row++)
  {
    for (int col = 0; col < other.dimensions.cols; col++)
    {
      result.data[row][col] = data[row][col] * other(row, col);
    }
  }
  return result;
}

// Calculate the L2 norm of the matrix
float Matrix::norm() const
{
  float sum = 0.0F;
  for (int row = 0; row < dimensions.rows; row++)
  {
    for (int col = 0; col < dimensions.cols; col++)
    {
      sum += (data[row][col] * data[row][col]);
    }
  }
  return std::sqrt(sum);
}

// Find the index of the maximum element in the matrix
int Matrix::argmax() const
{
  int max_index = 0;
  float max_value = data[0][0];

  for (int row = 0; row < dimensions.rows; row++)
  {
    for (int col = 0; col < dimensions.cols; col++)
    {
      if (data[row][col] > max_value)
      {
        max_value = data[row][col];
        max_index = col + (row * dimensions.cols);
      }
    }
  }

  return max_index;
}

// Matrix addition operator
Matrix Matrix::operator+(const Matrix& other) const
{
  if (dimensions.cols != other.dimensions.cols ||
      dimensions.rows != other.dimensions.rows) {
    throw std::length_error("The matrix size is not valid.");
  }

  Matrix add(dimensions);
  for (int row = 0; row < dimensions.rows; row++)
  {
    for (int col = 0; col < dimensions.cols; col++)
    {
      add.data[row][col] = data[row][col] + other.data[row][col];
    }
  }
  return add;
}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other)
{
  if (this != &other)
  {
    for (int row = 0; row < dimensions.rows; row++)
    {
      delete[] data[row];
    }
    delete[] data;

    dimensions = other.dimensions;
    data = new float*[dimensions.rows];
    for (int row = 0; row < dimensions.rows; row++)
    {
      data[row] = new float[dimensions.cols];
      for (int col = 0; col < dimensions.cols; col++)
      {
        data[row][col] = other.data[row][col];
      }
    }
  }
  return *this;
}

// Matrix multiplication with another matrix
Matrix Matrix::operator*(const Matrix& other) const
{
  if (dimensions.cols != other.dimensions.rows) {
    throw std::length_error("the multi is not defined.");
  }
  matrix_dims mult_dims;
  mult_dims.rows = dimensions.rows;
  mult_dims.cols = other.dimensions.cols;
  Matrix mult(mult_dims);
  for (int row = 0; row < mult.dimensions.rows; row++)
  {
    for (int col = 0; col < mult.dimensions.cols; col++)
    {
      float sum = 0.0F;
      for (int col_other = 0; col_other < dimensions.cols; col_other++)
      {
        sum += data[row][col_other] * other.data[col_other][col];
      }
      mult.data[row][col] = sum;
    }
  }
  return mult;
}

// Matrix multiplication with a scalar
Matrix Matrix::operator*(float scalar) const
{
  Matrix scal(dimensions);
  for (int row = 0; row < dimensions.rows; row++)
  {
    for (int col = 0; col < dimensions.cols; col++)
    {
      scal.data[row][col] = data[row][col] * scalar;
    }
  }
  return scal;
}

// Access individual elements of the matrix using parentheses (read/write)
float& Matrix::operator()(int row, int col) const
{
  if (((row >= 0) && (row < dimensions.rows)) || ((col >= 0) && col <
                                                                dimensions.cols))
  {
    return data[row][col];
  }
  else
  {
    throw std::out_of_range("Invalid matrix index.");
  }
}

// Access individual elements of the matrix using square brackets (read/write)
float& Matrix::operator[](int index) const
{
  if (index >= 0 && index < dimensions.rows * dimensions.cols)
  {
    int row_num = index / dimensions.cols;
    int col_num = index % dimensions.cols;
    return data[row_num][col_num];
  }
  else
  {
    throw std::out_of_range("Invalid matrix index.");
  }
}

// Output stream operator for easy printing of the matrix
std::ostream& operator<<(std::ostream& os, const Matrix& mat)
{
  for (int i = 0; i < mat.dimensions.rows; ++i)
  {
    for (int j = 0; j < mat.dimensions.cols; ++j)
    {
      os << mat.data[i][j] << " ";
    }
    os << std::endl;
  }
  return os;
}

// Input stream operator for reading matrix from a file
std::istream& operator>>(std::istream& s, Matrix& mat)
{
  std::streamsize matrix_size = mat.dimensions.rows * mat.dimensions.cols;

  s.seekg(0, std::istream::end);
  std::streamsize file_length = s.tellg();
  s.seekg(0, std::istream::beg);

  if (file_length < static_cast<std::streamsize>(matrix_size) * static_cast<std::streamsize>(sizeof(float)))
  {
    throw std::length_error("File not big enough for Matrix.");
  }

  auto* buffer = new float[matrix_size];
  s.read(reinterpret_cast<char*>(buffer),
         static_cast<std::streamsize>(matrix_size) * static_cast<std::streamsize>(sizeof(float)));

  for (int i = 0; i < mat.dimensions.rows; ++i)
  {
    for (int j = 0; j < mat.dimensions.cols; ++j)
    {
      mat.data[i][j] = buffer[i * mat.dimensions.cols + j];
    }
  }

  delete[] buffer;

  return s;
}

// Addition assignment operator to add another matrix to the current matrix
Matrix& Matrix::operator+=(const Matrix &other)
{
  if (dimensions.cols != other.dimensions.cols ||
      dimensions.rows != other.dimensions.rows) {
    throw std::length_error("The matrix size is not valid.");
  }
  *this = *this + other;
  return *this;
}
