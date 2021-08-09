#ifndef CLASS_DEMO_H
#define CLASS_DEMO_H

#include "MLClassifier.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace ImageUtilities {
// Paths
const std::string kImageDirectory{"../images"};

// Utility functions
std::vector<std::string> ImageFiles();

}; // namespace ImageUtilities

template <class T> class MessageQueue {
public:
  MessageQueue() {}
  MessageQueue(const MessageQueue &) = delete;
  MessageQueue &operator=(const MessageQueue &) = delete;
  ~MessageQueue() {}
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

  ClassificationDemo(const ClassificationDemo &) = delete;
  ClassificationDemo(ClassificationDemo &&) = delete;
  ClassificationDemo &operator=(const ClassificationDemo &) = delete;

  // getters / setters

  // typical behaviour methods
  void RunDemo();
  bool IsResultAvailable();
  std::tuple<std::string, std::string, double> GetNextResult();
  void Stop();

private:
  // typical behaviour methods
  void Run();
  /* Takes an int& so that the position within the images vector can be managed
   * and that state can be shared with the caller. */
  void AddClassification(int &);

  // member variables
  std::vector<std::string> _imageFiles;
  std::thread _demoThread;
  MessageQueue<std::tuple<std::string, std::string, double>> _msgQ;
  std::unique_ptr<MLClassifier> _classifier;
  std::mutex _mutex;
  std::atomic<bool> _stopped;
};

#endif