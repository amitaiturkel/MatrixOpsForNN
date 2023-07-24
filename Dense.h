#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"

typedef Matrix (*ActivationFunction)(const Matrix&);
class Dense{
  Matrix weight ;
  Matrix bi ;
  ActivationFunction function ;
 public:
  Dense(Matrix weights , Matrix  bias, ActivationFunction
  func );
  Matrix get_weights() const;
  Matrix get_bias() const;
  ActivationFunction get_activation() const;
  Matrix operator()(Matrix input) const;

};

#endif //DENSE_H
