%module streamdm

%{
    #define SWIG_FILE_WITH_INIT
    #include "streamdm.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

// Apply typemaps
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* samples, int nSampels, int nFeatures)};
%apply (int* IN_ARRAY1, int DIM1) {(int* targets, int nTargets)};
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* predictions, int nPredictions)};

// Rewrite methods
%feature("shadow") predict(double*, int, int, int*, int) %{
def predict(self, samples):
    predictions_len = len(samples)
    return $action(self, samples, predictions_len)
%}

%include "streamdm.h"

// Instantiate templates
%template(HoeffdingTree) LearnerWrapper<HT::HoeffdingTree>;
%template(HoeffdingAdaptiveTree) LearnerWrapper<HT::HoeffdingAdaptiveTree>;
%template(NaiveBayes) LearnerWrapper<NaiveBayes>;
%template(LogisticRegression) LearnerWrapper<LogisticRegression>;
%template(MajorityClass) LearnerWrapper<MajorityClass>;
%template(Perceptron) LearnerWrapper<Perceptron>;
%template(Bagging) LearnerWrapper<Bagging>;
