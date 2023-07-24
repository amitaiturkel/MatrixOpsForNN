#include "MlpNetwork.h"

digit MlpNetwork::operator() (const Matrix& input){
  Matrix m1 = input;
  m1 = layer_1(m1.vectorize());
  m1 = layer_2(m1);
  m1 = layer_3(m1);
  m1 = layer_4(m1);
  digit prob;
  prob.value = m1.argmax();
  prob.probability = m1(prob.value,0);
  return prob;

}