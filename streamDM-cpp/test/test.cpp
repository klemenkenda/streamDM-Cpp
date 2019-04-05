#include <iostream>
#include <fstream>
#include <typeinfo>
#include "Common.h"
#include "streams/ArffReader.h"
#include "streams/CSVReader.h"
#include "learners/Classifiers/Trees/HoeffdingTree.h"
#include "evaluation/BasicClassificationEvaluator.h"

using namespace std;
using namespace HT;

//int main(int argc, char* argv[]) {
//	string taskName;
//	string taskParams;
//	bool ret = CommandLineParser::parser(argc, argv, taskName, taskParams);
//	if (!ret) {
//		return 0;
//	}
//
//	Task* task = (Task*)CREATE_CLASS(taskName);
//	task->setParams(taskParams);
//	task->doTask();
//
//	return 0;
//}

int main()
{
	//string dataFile = "covtypeNorm.arff";
	string dataFile = "data.csv";
	//ArffReader reader;
	//CSVReader reader;

	//if (!reader.setFile(dataFile)) {
	//	printf("Failed to open file: %s ", dataFile.c_str());
	//	return 1;
	//}

	int counter = 0;
	//bool testing = false;
	int setSize = 10000; // max: 581012
	const double trainingSetFraction = 1.0;

	// Build vectors
	std::cout << "Building vectors... " << endl;

	vector<vector<double>> X;
	vector<int> y;
	//Instance* inst;
	
	//while (reader.hasNextInstance()) {
	//	counter++;
	//	printf("Counetr: %d\n", counter);

	//	if (counter > setSize) {
	//		break;
	//	}

	//	inst = reader.nextInstance();

	//	vector<double> x;

	//	for (int i = 0; i < inst->getNumberInputAttributes(); i++) {
	//		x.push_back(inst->getInputAttributeValue(i));
	//		//printf("%f ", inst->getInputAttributeValue(i));
	//	}

	//	X.push_back(x);
	//	y.push_back(inst->getOutputAttributeValue(0));
	//	//printf("%f\n", inst->getOutputAttributeValue(0));

	//	//cout << "Class: " << inst->getOutputAttributeValue(0) << " " << typeid(inst->getOutputAttributeValue(0)).name() << endl;
	//}

	ifstream infile(dataFile);

	if (infile.fail()) {
		printf("Failed to open file: %s ", dataFile.c_str());
		return 1;
	}

	string line;

	while (getline(infile, line)) {
		counter++;

		if (counter > setSize) {
			break;
		}

		istringstream iss(line);
		string str;

		if (!(iss >> str)) {
			break;
		}

		//cout << str << endl;

		const string delim = ",";
		size_t start = 0;
		size_t end = str.find(delim);
		vector<double> x;

		while ((end = str.find(delim, start)) != string::npos) {
			x.push_back(stod(str.substr(start, end - start)));
			//printf("%f ", stod(str.substr(start, end - start)));
			start = end + delim.size();
		}
		y.push_back(stod(str.substr(start, end)));
		//printf("%f\n", stod(str.substr(start, end)));
		X.push_back(x);
	}

	// Train
	std::cout << endl << "Training... " << endl;

	//vector<string> attribs {
	//	"",
	//	"",
	//	"",
	//	"",
	//	"",
	//	"",
	//	"",
	//	"",
	//	"",
	//	"",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//	"0,1",
	//};
	//vector<string> classes {
	//	"1,2,3,4,5,6,7"
	//};
	vector<string> attribs {
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
	};
	vector<string> classes {
		"0,1"
	};

	HoeffdingTree learner(attribs, classes);
	//learner.setInstanceInformation(attribs, classes);
	learner.setParams(
		"{"
			"\"MaxByteSize\":33554432,"
			"\"MemoryEstimatePeriod\":1000000,"
			"\"GracePeriod\":200,"
			"\"SplitConfidence\":0.0000001,"
			"\"TieThreshold\":0.05,"
			"\"BinarySplits\":false,"
			"\"StopMemManagement\":false,"
			"\"RemovePoorAtts\":false,"
			"\"LeafLearner\":\"NB\","
			"\"BbThreshold\":0,"
			"\"ShowTreePath\":false,"
			"\"TreePropertyIndexList\":\"\","
			"\"NoPrePrune\":false"
		"}"
	);
	learner.fit(X, y);

	// Predict
	std::cout << endl << "Predicting... " << endl;

	for (int i = 0; i < min(500, (int)X.size()); i++) {
		//printf("\n");
		//for (int j = 0; j < X[i].size(); j++) {
		//	printf("%f ", X[i][j]);
		//}
		//printf("%f ", (double)y[i]);
		std::cout << "Predicted: " << learner.predict(X[i]) << ", Expected: " << y[i] << endl;
	}

	// Output model
	string modelFile = "hoeffdingTree.json";

	if (!learner.exportToFile(modelFile)) {
		printf("Failed to export model to file: %s .", modelFile.c_str());
		return 1;
	}
	else {
		printf("Model written to: %s\n", modelFile.c_str());
	}

	return 0;
}
