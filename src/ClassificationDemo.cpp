#include "ClassificationDemo.h"
#include <filesystem>

// Utilitiy function for finding images files.
std::vector<std::string> ImageUtilities::ImageFiles() {
  std::vector<std::string> files;
  for (auto const &p :
       std::filesystem::directory_iterator(ImageUtilities::kImageDirectory)) {
    if (!p.is_directory()) { // TODO: add file extension filter
      files.emplace_back(p.path());
    }
  }
  if (files.size() < 1) {
    throw std::logic_error("No image files found");
  }
  return files;
}

// MessageQueue implementation
template <typename T> T MessageQueue<T>::Receive() {
  std::unique_lock<std::mutex> uLck(_mtx);
  _cv.wait(uLck, [this] { return !_queue.empty(); });
  T msg = std::move(_queue.back());
  _queue.pop_back();
  return msg;
}

template <typename T> void MessageQueue<T>::Send(T &&msg) {
  std::lock_guard<std::mutex> lck(_mtx);
  _queue.emplace_back(std::move(msg));
  _cv.notify_one();
}

template <typename T> bool MessageQueue<T>::IsMessageAvailable() {
  std::lock_guard<std::mutex> lck(_mtx);
  return _queue.size() > 0;
}

// ClassificationDemo Implementation
ClassificationDemo::ClassificationDemo()
    : _imageFiles(ImageUtilities::ImageFiles()),
      _classifier(std::make_unique<MLClassifier>()), _stopped(true) {}

ClassificationDemo::~ClassificationDemo() {}

void ClassificationDemo::RunDemo() {
  std::lock_guard<std::mutex> lck(_mutex);
  if (_stopped) {
    _stopped = false;
    _demoThread = std::thread(&ClassificationDemo::Run, this);
  }
}

void ClassificationDemo::Run() {
  int curPos = 0;
  this->AddClassification(curPos);
  long cycleDuration = 5 * 1000; // duration in ms
  std::chrono::time_point<std::chrono::system_clock> lastUpdate =
      std::chrono::system_clock::now();
  while (!_stopped) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // if sufficient time has elasped.
    long timeSinceUpdate =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - lastUpdate)
            .count();
    if (timeSinceUpdate >= cycleDuration) {
      std::string imgFile = _imageFiles.at(curPos);
      this->AddClassification(curPos);
      lastUpdate = std::chrono::system_clock::now();
    }
  }
}

void ClassificationDemo::AddClassification(int &curPos) {
  static int nbrImgs = _imageFiles.size();
  std::string imgFile = _imageFiles.at(curPos);
  auto clsResult = _classifier->ClassifyImage(imgFile);
  auto msg = std::make_tuple(imgFile, clsResult.first, clsResult.second);
  _msgQ.Send(std::move(msg));
  // wrap back around to the first image
  if (++curPos == nbrImgs) {
    curPos = 0;
  }
}

bool ClassificationDemo::IsResultAvailable() {
  return _msgQ.IsMessageAvailable();
}

// caller's thread will block on this method.
std::tuple<std::string, std::string, double>
ClassificationDemo::GetNextResult() {
  return _msgQ.Receive();
}

void ClassificationDemo::Stop() {
  std::lock_guard<std::mutex> lck(_mutex);
  if (!_stopped) {
    _stopped = true;
    _demoThread.join();
  }
}