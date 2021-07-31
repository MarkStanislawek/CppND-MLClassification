#include "Graphics.h"
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

Graphics::Graphics() : _windowName("ML Image Classification"), _stopped(false) {
  cv::namedWindow(_windowName, cv::WINDOW_NORMAL);
}

Graphics::~Graphics() { cv::destroyWindow(_windowName); }

void Graphics::Start() { _displayThread = std::thread(&Graphics::Run, this); }

void Graphics::Run() {
  while (!_stopped) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::string imgFile;
    std::string classification;
    double probability;
    if (_clsDemo->IsResultAvailable()) {
      std::tie(imgFile, classification, probability) = _clsDemo->GetNextResult();
      this->Display(imgFile, classification, probability);
    }
  }
}

void Graphics::Stop() {
  _stopped = true;
  _displayThread.join();
}

void Graphics::Display(const std::string &imgFile,
                       const std::string &classification, double probability) {
  // format confidence in the classification for display
  std::ostringstream oss;
  oss << std::fixed;
  oss << std::setprecision(2);
  oss << "Pr: " << (probability * 100) << "%";

  // overlay classification on image.
  cv::Mat image = cv::imread(imgFile);
  cv::Mat imageWithText = image.clone();
  int x = 10, y = 25;
  cv::Scalar fontColor(0, 255, 0);
  float fontScale = .5;
  cv::putText(imageWithText, classification, cv::Point(x, y),
              cv::FONT_HERSHEY_DUPLEX, fontScale, fontColor, 2);
  cv::putText(imageWithText, oss.str(), cv::Point(x, y + 15),
              cv::FONT_HERSHEY_DUPLEX, fontScale, fontColor, 2);

  // display image to user
  cv::imshow(_windowName, imageWithText);
  cv::waitKey(66);
}

void Graphics::SetDemonstrator(ClassificationDemo *clsDemo) { _clsDemo = clsDemo; }