#ifndef ML_CLASSIFIER_H_
#define ML_CLASSIFIER_H_

#include <opencv2/dnn.hpp>
#include <string>
#include <utility>

namespace MLUtilities {
const std::string kModelDir{"../model/tf_inception_v3/"};
const std::string kTfModelFile{kModelDir + "inception_v3_2016_08_28_frozen.pb"};
const std::string kClassNamesFile{kModelDir + "imagenet_slim_labels.txt"};
} // namespace MLUtilities

class MLClassifier {
private:
  cv::dnn::Net _network;
  std::vector<std::string> _classNames;

  std::vector<std::string> ReadClassNames(const std::string &filename);
  void GetMaxClass(const cv::Mat &probBlob, int *classId, double *classProb);

public:
  MLClassifier();
  ~MLClassifier();

  std::pair<std::string, double>
  ClassifyImage(const std::string &imageFile);
};

#endif /* MLCLASSIFIER_H_ */