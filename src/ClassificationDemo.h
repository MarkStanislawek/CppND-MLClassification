#ifndef CLASS_DEMO_H
#define CLASS_DEMO_H

#include <thread>
#include <string>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include <tuple>
#include <atomic>
#include "MLClassifier.h"

namespace ImageUtilities {
// Paths
const std::string kImageDirectory{"../images"};

// Utility functions
std::vector<std::string> ImageFiles();

}; // namespace ImageUtilities

template <class T> class MessageQueue {
public:
  void Send(T &&t);
  T Receive();
  bool IsMessageAvailable();

private:
  std::deque<T> _queue;
  std::mutex _mtx;
  std::condition_variable _cv;
};


class ClassificationDemo {
public:
  // constructor / desctructor
  ClassificationDemo();
  ~ClassificationDemo();

  // getters / setters

  // typical behaviour methods
  void RunDemo();
  bool IsResultAvailable();
  std::tuple<std::string,std::string,double> GetNextResult();
  void Stop();

private:
  // typical behaviour methods
  void Run();
  void AddClassification(int &);

  // member variables
  std::vector<std::string> _imageFiles;
  std::thread _demoThread;
  MessageQueue<std::tuple<std::string,std::string,double>> _msgQ;
  std::atomic<bool> _stopped;
  std::unique_ptr<MLClassifier> _classifier;
};

#endif