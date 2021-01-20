#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include "neuralnetwork.h"
using namespace std;




/////////////////////////////////// utils //////////////////////////////////////////

double sigmoid(double x){
	return 1/(1+exp(-x));
}

double derivative_sigmoid(double x){
	return sigmoid(x)*(1-sigmoid(x));
}

double inverse_sigmoid(double x){ ////////// not good but cba to cache
	return log(x/(1-x));
}

double init_random(){
	int sign = rand() % 2;
	double temp = (double)(rand() % 6) / 10.0;
    if (sign == 1) {
		temp = - temp;
	}
	return temp;
	//return (double) ((rand() % 21)-10)/10; // produces simple uniform distribution -1 to 1
}


///////////////////////////////// network class ////////////////////////////////////

neuralNetwork::neuralNetwork(int number, int* arr){
	numLayers = number;
	layerSizes = arr;
}

void neuralNetwork::about(){
	std::cout<<"number of layers: "<<numLayers<<endl;
	std::cout<<"layer sizes: {";
	for (int i=0;i<numLayers;i++){
		std::cout<<layerSizes[i]<<',';
	}
	std::cout<<"}"<<endl;
}

void neuralNetwork::generateValues(){
	srand((unsigned) time(0)); // can enter seeds to create same network each time

	values = new double* [numLayers];
	deltaValues = new double* [numLayers];
	biases = new double* [numLayers-1];
	deltaBiases = new double* [numLayers-1];
	weights = new double** [numLayers-1];
	deltaWeights = new double** [numLayers-1];

	for (int i=0; i<numLayers-1;i++){
		values[i+1] = new double [layerSizes[i+1]];
		deltaValues[i+1] = new double [layerSizes[i+1]];
		biases[i] = new double [layerSizes[i+1]]; 
		deltaBiases[i] = new double [layerSizes[i+1]]; 
		weights[i] = new double* [layerSizes[i+1]];
		deltaWeights[i] = new double* [layerSizes[i+1]];

		for (int j=0; j<layerSizes[i+1];j++){

			biases[i][j] = init_random(); // filling array just created
			deltaBiases[i][j] = 0;
			weights[i][j] = new double [layerSizes[i]];
			deltaWeights[i][j] = new double [layerSizes[i]];

			for (int k=0;k<layerSizes[i];k++){
				weights[i][j][k] = init_random(); // filling array just created
				deltaWeights[i][j][k] = 0;
			}
			
		}
	}

}


void neuralNetwork::feedForward(double* inputLayer){ // each subsequent layers neurons = sigmoid(sum of (weights*prevVals) +bias)
	double sigma;
	values[0] = inputLayer;
	for (int layerNum=1;layerNum<numLayers;layerNum++){
		for (int neurNum=0; neurNum<layerSizes[layerNum];neurNum++){
			sigma = 0;
			for (int prevNeurNum=0; prevNeurNum<layerSizes[layerNum-1];prevNeurNum++){
				sigma = sigma+values[layerNum-1][prevNeurNum]*weights[layerNum-1][neurNum][prevNeurNum];
			}
			values[layerNum][neurNum] = sigmoid(sigma+biases[layerNum-1][neurNum]);
		}
	}
}

void neuralNetwork::printOutput(){
	std::cout<<"[";
	for (int i=0;i<layerSizes[numLayers-1];i++){
		std::cout<<values[numLayers-1][i]<<", ";
	}
	std::cout<<"]"<<endl;
}

int neuralNetwork::returnOutput(){ // only for mnist
	int biggest = 0;
	for (int i=0;i<layerSizes[numLayers-1];i++){
		if (values[numLayers-1][i] > values[numLayers-1][biggest]){
			biggest = i;
		}
	}
	return biggest;
}

double neuralNetwork::lossFunc(double* desiredOutput){ 
	double loss = 0;
	int layerSize = layerSizes[numLayers-1];
	for (int i=0;i<layerSize;i++){
		loss += pow((values[numLayers-1][i]-desiredOutput[i]),2);
	}
	return 0.5*loss;
}

void neuralNetwork::updateDeltaMatrix(double* desiredOutput){
	double cache;
	for (int layerNum=(numLayers-1);layerNum>0;layerNum--){
		if(layerNum!=1){
			for (int prevValNum=0; prevValNum<layerSizes[layerNum-1];prevValNum++){
				deltaValues[layerNum-1][prevValNum]=0;
			}
		}
		for (int neurNum=0;neurNum<layerSizes[layerNum];neurNum++){
			if (layerNum==(numLayers-1)){
				cache = (values[layerNum][neurNum]-desiredOutput[neurNum])
					    *(derivative_sigmoid(inverse_sigmoid(values[layerNum][neurNum])));
				deltaBiases[layerNum-1][neurNum] += cache;

			} else {
				cache = deltaValues[layerNum][neurNum]
					    *(derivative_sigmoid(inverse_sigmoid(values[layerNum][neurNum])));
				deltaBiases[layerNum-1][neurNum] += cache;
			}
			for (int weightNum=0;weightNum<layerSizes[layerNum-1];weightNum++){
				if (layerNum!=1){
					deltaValues[layerNum-1][weightNum]+=cache*weights[layerNum-1][neurNum][weightNum];
				}
				deltaWeights[layerNum-1][neurNum][weightNum] += cache*(values[layerNum-1][weightNum]);
				//std::cout<<"deltaWeight: "<<layerNum-1<<' '<<neurNum<<' '<<weightNum<<' '<<deltaWeights[layerNum-1][neurNum][weightNum]<<endl;
			}
		}
	}
}

void neuralNetwork::applyDeltaMatrix(int batchSize, float learningRate){
	learningRate = batchSize*learningRate;
	for (int layerNum=0;layerNum<numLayers-1;layerNum++){
		for (int neurNum=0; neurNum<layerSizes[layerNum+1];neurNum++){

			biases[layerNum][neurNum] -=learningRate*deltaBiases[layerNum][neurNum];
			deltaBiases[layerNum][neurNum] = 0;
			for (int prevNeurNum=0; prevNeurNum<layerSizes[layerNum];prevNeurNum++){
				weights[layerNum][neurNum][prevNeurNum] -= learningRate*deltaWeights[layerNum][neurNum][prevNeurNum];
				deltaWeights[layerNum][neurNum][prevNeurNum] = 0;
			}
		}
	}
}

void neuralNetwork::closeNetwork(){
	for (int i=0; i<numLayers-1;i++){
		for (int j=0; j<layerSizes[i+1];j++){
			delete[] weights[i][j], deltaWeights[i][j];
		}
		delete[] values[i+1], deltaValues[i+1];
		delete[] biases[i], deltaBiases[i];
		delete[] weights[i], deltaWeights[i];
	}
	delete[] values, deltaValues;
	delete[] biases, deltaBiases;
	delete[] weights, deltaWeights;
}

