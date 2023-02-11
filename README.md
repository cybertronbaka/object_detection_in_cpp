![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)

# Object Detection in C++ Reference 
Keeping this here because Documentation for TensorflowLite in C++ is like finding a needle in a haystack.

These are for personal reference and I am using the repo for personal notes. If you can follow the notes, then it might as well be yours.

***To Build and Run***
```sh
mkdir build && cd build
cmake ..
make
./object_detection ../efficientdet_lite0.tflite ../coco.names
```

***To Build and Run minimal.cpp***
```sh
g++ minimal.cpp -I/usr/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui --std=c++17 -I/usr/local/include/ -ltensorflowlite -o minimal

./minimal efficientdet_lite0.tflite
```

***To build and run io_tensors.cpp***
```sh
g++ io_tensors.cpp -I/usr/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui --std=c++17 -I/usr/local/include/ -ltensorflowlite -lopencv_imgcodecs -lopencv_imgproc -o io_tensors

./io_tensors efficientdet_lite0.tflite coco.names test.jpg
```


## Challenge 1: Installation
**(For Debian based Linux only)**
***If you have no idea what you are doing, like me, then it is a tiger level challenge.***
### Round 1: Install Opencv
Install minimal prerequisites
```sh
sudo apt update && sudo apt install -y cmake g++ wget unzip
```
Download and unpack source
```sh
mkdir opencv4 && cd opencv4
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
```
Create build directory
```sh
mkdir -p build && cd build
```
Cmake
```sh
cmake  ../opencv-4.x
```
build
```sh
make -j4
```
install
```sh
make install
```

### Round 2: Install Tensorflow lite
Install the numpy and keras_preprocessing
```sh
pip install numpy wheel
pip install keras_preprocessing --no-deps
```
Clone the tensorflow repository. 
```sh
git clone git@github.com:tensorflow/tensorflow.git
```

Check `.bazelversion` file and download that version of bazel.
Note: Bazel Cache is saved at ~/.cache. Just in case your system runs out of memory and bazel takes up too much of your space

Configure the and prepare for bazel build. You can select `N` for everyhing. However, be sure to read everything while configuring.
```sh
./configure
```

Build using bazel
```sh
bazel build -c opt //tensorflow/lite:libtensorflowlite.so
```
If you want to build for tensorflow
```sh
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so 
```


Here `opt` is for optimized build. After the build, `libtensorflowlite.so` will be generated and that will be required when compiling. So, copy this file `libtensorflowlite.so` to `/usr/local/lib` or wherever you please.
```sh
export tensorflow_root=your/desired/installation/path
cp -d bazel-bin/tensorflow/libtensorflowlite.so* $tensorflow_root/lib/
```
My `$tensorflow_root` was `/usr/local`

For Tensorflow:
```sh
cp -d bazel-bin/tensorflow/libtensorflow_cc.so* $tensorflow_root/lib/
cp -d bazel-bin/tensorflow/libtensorflow_framework.so* $tensorflow_root/lib/
cp -d $tensorflow_root/lib/libtensorflow_framework.so.2 $tensorflow_root/lib/libtensorflow_framework.so
```

Then copy the headers:
```sh
cp -r bazel-bin/ $tensorflow_root/include/
cp -r tensorflow/cc $tensorflow_root/include/tensorflow/
cp -r tensorflow/core $tensorflow_root/include/tensorflow/
cp -r tensorflow/lite $tensorflow_root/include/tensorflow/
cp -r third_party/ $tensorflow_root/include/third_party/
cp -r bazel-tensorflow/external/eigen_archive/Eigen/ $tensorflow_root/include/Eigen/
cp -r bazel-tensorflow/external/eigen_archive/unsupported/ $tensorflow_root/include/unsupported/
cp -r bazel-tensorflow/external/com_google_protobuf/src/google/ $tensorflow_root/include/google/
cp -r bazel-tensorflow/external/com_google_absl/absl/ $tensorflow_root/include/absl/
```

## Challenge 2: Basic Inference Workflow
**Not even a challenge**

***Note:  Often, when you switch your model you will need to update the preprocess and postprocess stages. As in, in the pre-process stage, you will need to change the type of input tensor or you will need to resize your input data and convert your data into a tensor. In the post-process stage, you will need to figure out which ones are the output layers in the NN through minimal.cpp or you will need to figure out how many output tensors are there, what are each tensors for and how to post-process each one of them.***
1. Load the TensorFlow Lite model: You can load the TensorFlow Lite model using the tflite::FlatBufferModel class, which provides methods to parse the model from a file or a buffer in memory.
2. Create an interpreter: An interpreter is an instance of the tflite::Interpreter class, which provides methods for executing the model on input data. You can create an interpreter by passing the loaded model to its constructor.

3. Allocate tensors: Tensors are the data structures that hold the input and output data for the model. You can allocate tensors using the AllocateTensors method of the interpreter.

4. Pre-process the input image: You will typically need to pre-process the input image to resize it, normalize its pixel values, and prepare it for the model. To figure out the shape of the input tensor you can just run the compile and run minimal.cpp

5. Fill the input tensor: You can fill the input tensor with the pre-processed input image by copying the image data into the tensor's buffer.

6. Run the model: You can run the model on the input tensor by calling the Invoke method of the interpreter.

7. Post-process the output: The output tensor will contain the raw predictions from the model. You will typically need to post-process the output to extract the bounding boxes and class scores for the detected objects. ( Here, you will need to look inside the documentation of the pretrained model or you can figure it out by running minimal.cpp )

8. Display the results: Finally, you can display the results by drawing the bounding boxes and class labels on the input image using OpenCV's drawing functions.

# Links that could be useful
[How To Install Tensorflow for C++ | Tensorflow Tutorials](https://www.youtube.com/watch?v=He2p2JLpYC0)
[Download | CMake](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
[OpenCV: Installation in Linux](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
[How to build and use Google TensorFlow C++ api](https://stackoverflow.com/questions/33620794/how-to-build-and-use-google-tensorflow-c-api)
[.so, .a, .dll Differences](https://stackoverflow.com/questions/9688200/difference-between-shared-objects-so-static-libraries-a-and-dlls-so)
[CMake Tutorial](https://www.youtube.com/watch?v=nlKcXPUJGwA&list=PLalVdRk2RC6o5GHu618ARWh0VO0bFlif4&index=1&t=0s)
[OpenCV C++ tutorial](https://www.youtube.com/watch?v=uJrwLq_BKPY&list=PLkmvobsnE0GHMmTF7GTzJnCISue1L9fJn)

#### TODO:
1. Cross compile for arm based devices like RPI.
2. Test in RPI and benchmark.
3. See if this code can be deployed to a microcontroller like arduino/pico