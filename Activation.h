
#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "Matrix.h"
namespace activation {
    typedef Matrix (*ActivationFunction)(const Matrix&);

    Matrix relu( const Matrix& matrix_x );
    Matrix softmax(const Matrix& matrix_x);
}


#endif //ACTIVATION_H