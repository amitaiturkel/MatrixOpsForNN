//MlpNetwork.h

#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"

#define MLP_SIZE 4
#define ONE 1
#define TWO 2
#define THREE 3
#define ZERO 0

/**
 * @struct digit
 * @brief Identified (by Mlp network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit {
	unsigned int value;
	float probability;
} digit;

const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
									{64,  128},
									{20,  64},
									{10,  20}};
const matrix_dims bias_dims[] = {{128, 1},
								 {64,  1},
								 {20,  1},
								 {10,  1}};

class MlpNetwork {
 private:
  const Matrix weights[MLP_SIZE];
  const Matrix bias[MLP_SIZE];
  const Dense layer_1;
  const Dense layer_2;
  const Dense layer_3;
  const Dense layer_4;

 public:
  MlpNetwork(const Matrix weights[],const Matrix bias[]) :
      layer_1(weights[ZERO], bias[ZERO], activation::relu),
      layer_2(weights[ONE], bias[ONE], activation::relu),
      layer_3(weights[TWO], bias[TWO], activation::relu),
      layer_4(weights[THREE], bias[THREE], activation::softmax) {
  }
  digit operator() (const Matrix& input);

};

#endif // MLPNETWORK_H