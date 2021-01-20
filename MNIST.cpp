#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include "neuralnetwork.h"
using namespace std;

///////////////////////////////// relevant files /////////////////////////////////
string const image_file = "C:/Users/Archie/Documents/Project Dump/Simple Neural Network/train-images-idx3-ubyte/train-images.idx3-ubyte";
string const label_file = "C:/Users/Archie/Documents/Project Dump/Simple Neural Network/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
string const loss_file = "C:/Users/Archie/Documents/Project Dump/Simple Neural Network/lossData.txt";
string const test_image_file = "C:/Users/Archie/Documents/Project Dump/Simple Neural Network/t10k-images.idx3-ubyte";
string const test_label_file = "C:/Users/Archie/Documents/Project Dump/Simple Neural Network/t10k-labels.idx1-ubyte";
//"C:\Users\Archie\Documents\Project Dump\Simple Neural Network\t10k-labels.idx1-ubyte"
//////////////////////////////// hyper params ////////////////////////////////////
float const learningRate = 0.01;
int const batchSize = 10;


//////////////// about ///////////////
void about(neuralNetwork brain){
	std::cout << "***************** Simple Neural Network with stats *****************"<<endl;
	brain.about();
	std::cout << "learning rate: "<<learningRate<<endl<<"batchSize: "<<batchSize<<endl;
	std::cout << "********************* Start Training ********************"<<endl;
	system("pause");
}
/////////////////////////////////// main ////////////////////////////////////////////
int main () {
	int numLayers =4;
	int layerSizes[numLayers] = {784, 64, 64, 10};

	ifstream imageFile (image_file, ios::in|ios::binary);
	ifstream labelFile (label_file, ios::in|ios::binary);
	ofstream lossFile (loss_file);
	std::cout<<"pos 1;"<<endl;
	if (!imageFile or !labelFile){
		std::cout << "error opening file";
		return 1;
	}
	
	neuralNetwork brain(numLayers, layerSizes);
	brain.generateValues();
	double inputLayer[784];
	double desiredOutput[10] = {0,0,0,0,0,0,0,0,0,0};
	double* outputLayer;
	unsigned char pixInt;
	unsigned char labelInt;
	double loss;

	about(brain);


	for(int epochNum=0;epochNum<1;epochNum++){
		imageFile.seekg(16, ios::beg); // start of the images
		labelFile.seekg(8, ios::beg); 
		for(int picNum=0;picNum<6000;picNum++){
			loss = 0;
			for (int batchNum=0;batchNum<batchSize;batchNum++){
				
				for (int a=0;a<784;a++){
					imageFile.read ((char *) &pixInt, sizeof(char));
					inputLayer[a] = (((double) pixInt) *(0.99/255))+0.01;
				}
				labelFile.read((char *) &labelInt, sizeof(char));

				desiredOutput[(int) labelInt] = 1;
				brain.feedForward(inputLayer);
				loss+=brain.lossFunc(desiredOutput);
				std::cout<<loss<<endl;

				brain.updateDeltaMatrix(desiredOutput);
				brain.printOutput();
				std::cout<<"label:"<<(int) labelInt<<endl;
				desiredOutput[(int) labelInt] = 0;
			}
			brain.applyDeltaMatrix(batchSize, learningRate);
			lossFile<<loss/batchSize<<",";
			////////////////// print image
			for (int q=0;q<28;q++){
				for (int t=0;t<28;t++){
					if (inputLayer[28*q+t] == 0.01){
						std::cout<<0;
					} else{
						std::cout<<1;
					}
				}
				std::cout<<endl;
			}
			////////////////////////////
		}
	}
	imageFile.close();
	labelFile.close();
  	lossFile.close();

  	ifstream imageTestFile (test_image_file, ios::in|ios::binary);
	ifstream labelTestFile (test_label_file, ios::in|ios::binary);
	std::cout<<"pos 1;"<<endl;
	if (!imageTestFile or !labelTestFile){
		std::cout << "error opening file";
		return 1;
	}
	imageTestFile.seekg(16, ios::beg); // start of the images
	labelTestFile.seekg(8, ios::beg); 
	int totalCorrect = 0;
	for (int testImageNumber = 0; testImageNumber<10000;testImageNumber++){
		for (int a=0;a<784;a++){
			imageTestFile.read ((char *) &pixInt, sizeof(char));
			inputLayer[a] = (((double) pixInt) *(0.99/255))+0.01;
		}
		labelTestFile.read((char *) &labelInt, sizeof(char));
		brain.feedForward(inputLayer);
		if (labelInt == brain.returnOutput()){
			totalCorrect++;
		}
	}
	imageTestFile.close();
	labelTestFile.close();

  	std::cout<<totalCorrect<<"out of 10000"<<endl;
	brain.closeNetwork();
	return 0;
}