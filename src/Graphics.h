#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "ClassificationDemo.h"
#include <atomic>
#include <thread>
#include <mutex>



class Graphics {
public:
  // constructor / desctructor
  Graphics();
  ~Graphics();

  Graphics(const Graphics&) = delete;
  Graphics(Graphics&&) = delete;
  Graphics& operator=(const Graphics&) = delete;

  // getters / setters
  void SetDemonstrator(ClassificationDemo *);

  // typical behaviour methods
  void Start();
  void Stop();

private:
  // typical behaviour methods
  void Display(const std::string &imgFile, const std::string &classification, double probability);
  void Run();

  // member variables
  std::string _windowName;
  ClassificationDemo *_clsDemo;
  std::atomic<bool> _stopped;
  std::thread _displayThread;
  std::mutex _mutex;
};

#endif