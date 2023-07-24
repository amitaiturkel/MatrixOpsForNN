
  #ifndef MATRIX_H
  #define MATRIX_H

#include <iostream>
#include <cmath>
#include <stdexcept>
  /**
   * @struct matrix_dims
   * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
   */
  typedef struct matrix_dims
  {
      int rows, cols;
  } matrix_dims;

  class Matrix
  {
   private:
    matrix_dims dimensions;
    float **data;

   public:
    Matrix ();
    Matrix (matrix_dims dims);
    Matrix ( int rows,  int cols);
    Matrix (const Matrix &other) noexcept;
    ~Matrix ();
    int get_rows () const;
    int get_cols () const;
    float sum() const;
    Matrix &transpose ();
    Matrix &vectorize ();
    void plain_print () const;
    Matrix dot (const Matrix &other) const;
    float norm () const;
    int argmax () const;
    Matrix operator+ (const Matrix &other) const;
    Matrix& operator+= ( const Matrix &other) ;
    Matrix &operator= (const Matrix &other);
    Matrix operator* (const Matrix &other) const;
    Matrix operator* ( float scalar) const;
    float &operator() ( int row,  int col) const;
    float &operator[] ( int index) const;
    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);
    friend std::istream& operator>> (std::istream &is, Matrix& m);
  };
#endif // MATRIX_H
