//
// Created by punims on 12/12/2019.
//


// Make sure all function names exist by trying to call and compile them:

#include "Activation.cpp"
#include "MlpNetwork.cpp"
#include "Matrix.cpp"
#include "Dense.cpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#define QUIT "q"
#define INSERT_IMAGE_PATH "Please insert image path:"
#define ERROR_INAVLID_PARAMETER "Error: invalid Parameters file for layer: "
#define ERROR_INVALID_INPUT "Error: Failed to retrieve input. Exiting.."
#define ERROR_INVALID_IMG "Error: invalid image path or size: "
#define USAGE_MSG "Usage:\n" \
				  "\t./mlpnetwork presubmit.inw1 presubmit.inw2 presubmit.inw3 presubmit.inw4 presubmit.inb1 presubmit.inb2 presubmit.inb3 presubmit.inb4\n" \
				  "\twi - the i'th layer's weights\n" \
				  "\tbi - the i'th layer's biases"

#define ARGS_START_IDX 1
#define ARGS_COUNT (ARGS_START_IDX + (MLP_SIZE * 2) + 1)
#define WEIGHTS_START_IDX ARGS_START_IDX
#define BIAS_START_IDX (ARGS_START_IDX + MLP_SIZE)

// FORWARD DECLARATION:
int compile_matrix();
void compile_activation();
void compile_dense();

/**
 * Prints program usage to stdout.
 */
void usage()
{
	std::cout << USAGE_MSG << std::endl;
}

/**
 * Given a binary file path and a matrix,
 * reads the content of the file into the matrix.
 * file must match matrix in size in order to read successfully.
 * @param filePath - path of the binary file to read
 * @param mat -  matrix to read the file into.
 * @return boolean status
 *          true - success
 *          false - failure
 */
bool readFileToMatrix(const std::string& filePath, Matrix& mat)
{
	std::ifstream is;
	is.open(filePath, std::ios::in | std::ios::binary);
	if (!is.is_open())
	{
		return false;
	}

	is >> mat;
	is.close();
	return true;
}

/**
 * Loads MLP paraemters from weights & biases paths
 * to Weights[] and Biases[].
 * Exits (code == 1) upon failures.
 * @param paths array of programs arguments, expected to be mlp parameters
 *        path.
 * @param weights array of matrix, weigths[i] is the i'th layer weights matrix
 * @param biases array of matrix, biases[i] is the i'th layer bias matrix
 *          (which is actually a vector)
 */
void loadParameters(std::string paths[ARGS_COUNT],
	Matrix weights[MLP_SIZE], Matrix biases[MLP_SIZE])
{
	for (int i = 0; i < MLP_SIZE; i++)
	{
		weights[i] = Matrix(weights_dims[i].rows, weights_dims[i].cols);
		biases[i] = Matrix(bias_dims[i].rows, bias_dims[i].cols);
		std::string weightsPath(paths[WEIGHTS_START_IDX + i]);
		std::string biasPath(paths[BIAS_START_IDX + i]);
		if (!(readFileToMatrix(weightsPath, weights[i]) &&
			readFileToMatrix(biasPath, biases[i])))
		{
			std::cerr << ERROR_INAVLID_PARAMETER << (i + 1) << std::endl;
			exit(EXIT_FAILURE);
		}

	}
}

/**
 * This programs Command line interface for the mlp network.
 * Looping on: {
 *                  Retrieve user input
 *                  Feed input to mlpNetwork
 *                  print image & netowrk prediction
 *             }
 * Exits (code == 1) on fatal errors: unable to read user input path.
 * @param std::ifstream& input to use in order to predict img.
 * @param MlpNetwork &mlp to use in order to predict img.
 * @param std::string[] list_of_files to use in order to predict img.
 *
 */
void mlpCli(MlpNetwork& mlp)
{
	Matrix img(img_dims.rows, img_dims.cols);
	std::string imgPath;
	for (int i = 0; i < 1; ++i)
	{
		if (readFileToMatrix("presubmit.inim" + std::to_string(i), img))
		{
			const Matrix imgVec = img;
			digit output = mlp(imgVec);
			std::cout << "Image processed:" << std::endl
				<< img << std::endl;
			std::cout << "Mlp result: " << output.value <<
				" at probability: " << output.probability << std::endl;
			assert(output.value == 5);
			assert(output.probability > 0.9);
		}
		else
		{
			std::cout << ERROR_INVALID_IMG << imgPath << std::endl;
			break;
		}
		if (i == 9)
		{
			std::cout << "Done reading Inputs" << std::endl;
		}
	}

}

int compile_matrix()
/***
 * This function checks for the existence of all matrix functions
 * and some functionality.
 * @return
 */
{
	std::cout << "Checking Matrix class:" << std::endl;
	std::cout << "\tcreating a 3x3 matrix" << std::endl;

	std::cout << "\tconstructors" << std::endl;
	// check ctor
	Matrix m(3, 3);
	const Matrix m2(3, 3);
	for (int i = 0; i < 9; i++)
	{
		assert(m2[i] == 0);
	}

	// check default ctor
	const Matrix null;
	assert(null[0] == 0);

	// Check copy ctor
	Matrix b(null);

	std::cout << "\toperators () and []" << std::endl;
	for (int i = 0; i < m.get_rows(); ++i)
	{
		for (int j = 0; j < m.get_cols(); ++j)
		{
			m(i, j) = (float)(i + j);
		}
	}
	for (int i = 0; i < m.get_cols() * m.get_rows(); i++)
	{
		m[i] = (float)i; // check [] operator
	}

	std::cout << "\tplain_print, expected:" << std::endl;
	// check plain print
	std::cout << "0 1 2 \n3 4 5 \n6 7 8 " << std::endl;
	std::cout << "\tgot:" << std::endl;
	const Matrix m3(m);
	m3.plain_print();

	std::cout << "\tm.get_rows() -> " << m.get_rows() << std::endl;
	// check default ctor constructs as instructed
	assert(m3.get_rows() == 3);

	std::cout << "\tm.get_rows() -> " << m.get_cols() << std::endl;
	// check default ctor constructs as instructed
	assert(m3.get_rows() == 3);
	Matrix old_m = m3;

	// check sum operator
	std::cout << "\tm.sum() -> " << m3.sum() << std::endl;
	assert(m3.sum() == 36);

	// check norm operator
	std::cout << "\tm.norm() -> " << m3.norm() << std::endl;
	assert(m3.norm() > 14 && m3.norm() < 15);

	// check argmax
	std::cout << "\tm.argmax() -> " << m3.argmax() << std::endl;
	assert(m3.argmax() == 8);

	// check dot product between matrices
	std::cout << "\tm.dot(m) ->" << std::endl;
	const Matrix d = m3.dot(m3);
	d.plain_print();
	assert(d(1, 1) == 16);

	//check mult
	std::cout << "\tm * m -> " << std::endl;
	const Matrix mult1 = m3 * m3;
	mult1.plain_print();
	assert(mult1(1, 1) == 54);

	//check mult
	std::cout << "\tm * 0.5 -> " << std::endl;
	const Matrix mult2 = m3 * 0.5;
	mult2.plain_print();
	assert(mult2(1, 1) == 2);

	//check mult
	std::cout << "\t1.5 * m -> " << std::endl;
	const Matrix mult3 = m3 * 1.5;
	mult3.plain_print();
	assert(mult3(1, 1) == 6);

	//check addition
	std::cout << "\tm + m -> " << std::endl;
	const Matrix add1 = m3 + m3;
	add1.plain_print();
	assert(add1(1, 1) == 8);

	// check transpose operator
	std::cout << "\tm.vectorize()\n\tm.transpose() -> ";
	std::cout.flush();
	m.vectorize().transpose().vectorize().transpose()
		.transpose().transpose().plain_print();
	assert((m.get_cols() == 9) && (m.get_rows() == 1));


	// Check = operator

	std::cout << "\tnew_mat = m -> ";
	std::cout.flush();
	(b = m3).plain_print();
	for (int i = 0; i < m3.get_cols() * m3.get_rows(); i++)
	{
		assert(b[i] == m3[i]);
	}

	//check self addition
	std::cout << "\tm += m -> ";
	std::cout.flush();
	m += m;
	m.plain_print();
	assert(m(0, 4) == 8);

	const Matrix m4(m);

	m += m4;

	// operator << and operator >> are checked in the main running this function

	std::cout << "Passed: All Matrix functions exist" << std::endl
		<< std::endl;
	return 1;
}

void compile_activation()
/**
 * function which checks the existence of all functions from the activation class
 * @return
 */
{
	std::cout << "Checking namespace activation:" << std::endl;

	std::cout << "\tLooking for functions... ";
	// check Activation functions
	(void)activation::relu;
	(void)activation::softmax;

	std::cout << "found!" << std::endl;

	Matrix m(1, 9);
	for (int i = 0; i < m.get_cols() * m.get_rows(); i++)
	{
		m[i] = (float)(i - 2);
	}
	std::cout << "\n\tFor m = ";
	std::cout.flush();
	m.plain_print();
	const Matrix m1(m);
	std::cout << "\n\tactivation::relu(m) -> ";
	std::cout.flush();
	const Matrix new_m1 = activation::relu(m1);
	new_m1.plain_print();
	assert(new_m1(0, 1) == 0);

	std::cout << "\tactivation::softmax(m) -> ";
	std::cout.flush();
	const Matrix new_m2 = activation::softmax(m1);
	new_m2.plain_print();
	assert(std::abs(new_m2(0, 8) - 0.632199) < 0.1);
	assert(std::abs(new_m2.sum() - 1) < 0.1);

	std::cout << "Passed: All Activation functions exist" << std::endl
		<< std::endl;

}

void compile_dense()
/**
 * function to check the existence of all dense class methods
 */
{
	std::cout << "Checking Dense class:" << std::endl;
	Matrix m;
	Matrix bias;
	Matrix w;
	m[0] = 2;
	bias[0] = 1;
	w[0] = 3;

	Matrix m1(m);
	Matrix w1(w);
	Matrix bias1(bias);


	std::cout << "\tconstructor" << std::endl;
	// check ctor of Dense
	 Dense d(w1, bias1, activation::relu);

	std::cout << "\tget_weights, get_bias, get_activation" << std::endl;
	//checking existence of getWeights and getBias
	d.get_weights();
	d.get_bias();
	d.get_activation();

	// check () operator
	std::cout << "\toperator ()" << std::endl;
	const Matrix D = d(m1);
	assert(D[0] == 7);
	std::cout << "Passed: All Dense functions exist" << std::endl << std::endl;
}

/**
 * Program's main
 * @param argc count of args
 * @param argv args values
 * @return program exit status code
 */
int main()
{
	std::cout << "Remember! This is only a basic test." << std::endl;
	std::cout << "Checking functions exist and basic functionality:"
		<< std::endl;
	compile_matrix();
	compile_activation();
	compile_dense();
	// std:: cout << argc << " " << ARGS_COUNT << std::endl;
	// if(argc != ARGS_COUNT){
	// 	usage();
	// 	exit(EXIT_FAILURE);
	// }
	Matrix weights[MLP_SIZE];
	Matrix biases[MLP_SIZE];
	std::string argv[9] = { "stub_do_not_delete", "presubmit.inw1", "presubmit.inw2",
						   "presubmit.inw3", "presubmit.inw4",
						   "presubmit.inb1", "presubmit.inb2", "presubmit.inb3",
						   "presubmit.inb4" };
	loadParameters(argv, weights, biases);

	std::cout << "Checking MlpNetwork, displaying output:" << std::endl;

	Matrix  weights1[] = { weights[0], weights[1], weights[2], weights[3] };
	Matrix  biases1[] = { biases[0], biases[1], biases[2], biases[3] };

	MlpNetwork mlp(weights1, biases1);

	mlpCli(mlp);

	std::cout << "All presubmit tests finished!" << std::endl;
	return EXIT_SUCCESS;
}



//mlpNetowrk is tested in the main.


