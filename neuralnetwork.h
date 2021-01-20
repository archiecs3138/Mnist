#ifndef neuralnetwork
#define neuralnetwork

////////// utils ///////////
double sigmoid(double x);
double derivative_sigmoid(double x);
double inverse_sigmoid(double x); ////////// this is terrible get rid asap
double init_random();
////////// class ///////////
class neuralNetwork{
	protected:	    
		int numLayers;
		int *layerSizes;

        double *** weights; // array of arrays of arrays
        double *** deltaWeights;

        double ** values; // array of arrays
        double ** deltaValues;

        double ** biases; // array of arrays
        double ** deltaBiases;

	public:
		neuralNetwork(int number, int* arr);
		void about();
		void generateValues();
		void feedForward(double* inputLayer);
		void printOutput();
		int returnOutput();
		double lossFunc(double* desiredOutput);
		void updateDeltaMatrix(double* desiredOutput);
		void applyDeltaMatrix(int batchSize, float learningRate);
		void closeNetwork();
};

#endif
