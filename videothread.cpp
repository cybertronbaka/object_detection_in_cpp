#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex videoThreadMutex;

class VideoThread {
 public:
  VideoThread() : running_(false), frame_ready_(false) {}

  // Start the video thread
  void Start() {
    running_ = true;
    video_thread_ = std::thread(&VideoThread::Run, this);
  }

  // Stop the video thread
  void Stop() {
    running_ = false;
    video_thread_.join();
  }

  // Get the latest frame
  cv::Mat GetFrame() {
    std::unique_lock<std::mutex> lock(videoThreadMutex);
    frame_ready_cv_.wait(lock, [this]{return frame_ready_;});
    return frame_;
  }

 private:
  // The video thread function
  void Run() {
    cv::VideoCapture cap(0, cv::CAP_ANY);
    while (running_) {
      cap >> frame_;
      {
        std::lock_guard<std::mutex> lock(videoThreadMutex);
        frame_ready_ = true;
      }
      frame_ready_cv_.notify_all();
    }
  }

  std::thread video_thread_;
  cv::Mat frame_;
  bool running_;
  bool frame_ready_;
  std::condition_variable frame_ready_cv_;
};