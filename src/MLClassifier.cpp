#include "MLClassifier.h"
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

MLClassifier::MLClassifier()
    : _network(cv::dnn::readNetFromTensorflow(MLUtilities::kTfModelFile)),
      _classNames(ReadClassNames(MLUtilities::kClassNamesFile)) {
  if (_network.empty())
    throw std::logic_error("Neural network is empty");
}

// Deep copy.  Will create another cv::dnn::Net instance.
MLClassifier::MLClassifier(const MLClassifier &mlc)
    : _network(mlc._network), _classNames(mlc._classNames) {}

MLClassifier::MLClassifier(MLClassifier &&mlc)
    : _network(std::move(mlc._network)),
      _classNames(std::move(mlc._classNames)) {}

// Shallow copy assignment.
MLClassifier &MLClassifier::operator=(const MLClassifier &mlc) {
  if (this == &mlc)
    return *this;
  _network = mlc._network;
  _classNames = mlc._classNames;
  return *this;
}

MLClassifier::~MLClassifier() {}

std::pair<std::string, double>
MLClassifier::ClassifyImage(const std::string &imageFile) {
  cv::Mat img = cv::imread(imageFile);
  if (img.empty()) {
    throw std::logic_error("Image file is empty");
  }

  // Adjust the image to match training settings for inception v3.
  cv::Size s(299, 299);
  cv::Mat normalImg;
  cv::resize(img, img, s, 0, 0, cv::INTER_CUBIC);
  // Convert the image to float and normalize data:
  img.convertTo(normalImg, CV_32FC1);
  // normalize
  normalImg -= 0;
  normalImg /= 255;
  cv::Mat inputBlob = cv::dnn::blobFromImage(normalImg);

  _network.setInput(inputBlob);

  cv::Mat result = _network.forward();

  int classId;
  double classProb;
  // find the best classification
  GetMaxClass(result, &classId, &classProb);

  return std::make_pair(_classNames.at(classId), classProb);
}

std::vector<std::string>
MLClassifier::ReadClassNames(const std::string &filename) {
  std::vector<std::string> classNames;

  std::ifstream fp(filename);
  if (!fp.is_open())
    throw std::logic_error("Cannot open class names file.");

  std::string name;
  while (!fp.eof()) {
    std::getline(fp, name);
    if (name.length())
      classNames.push_back(name);
  }

  fp.close();
  return classNames;
}

void MLClassifier::GetMaxClass(const cv::Mat &probBlob, int *classId,
                               double *classProb) {
  cv::Mat probMat = probBlob.reshape(1, 1); // reshape the blob to 1x1000 matrix
  cv::Point classNumber;
  cv::minMaxLoc(probMat, 0, classProb, 0, &classNumber);
  *classId = classNumber.x;
}