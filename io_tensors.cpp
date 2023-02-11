#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <vector>

std::string GetTensorTypeString(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
        return "kTfLiteFloat32";
    case kTfLiteInt32:
        return "kTfLiteInt32";
    case kTfLiteUInt8:
        return "kTfLiteUInt8";
    case kTfLiteInt64:
        return "kTfLiteInt64";
    case kTfLiteString:
        return "kTfLiteString";
    case kTfLiteBool:
        return "kTfLiteBool";
    case kTfLiteInt16:
        return "kTfLiteInt16";
    case kTfLiteComplex64:
        return "kTfLiteComplex64";
    case kTfLiteInt8:
        return "kTfLiteInt8";
    case kTfLiteFloat16:
        return "kTfLiteFloat16";
    case kTfLiteFloat64:
        return "kTfLiteFloat64";
    case kTfLiteComplex128:
        return "kTfLiteComplex128";
    case kTfLiteUInt64:
        return "kTfLiteUInt64";
    case kTfLiteResource:
        return "kTfLiteResource";
    case kTfLiteVariant:
        return "kTfLiteVariant";
    case kTfLiteUInt32:
        return "kTfLiteUInt32";
    case kTfLiteUInt16:
        return "kTfLiteUInt16";
    case kTfLiteInt4:
        return "kTfLiteInt4";
    default:
         return "Unknown Tensor Type";
  }
}

cv::Mat image;

void onMouse(int event, int x, int y, int flags, void* param){
    char text[100];
    cv::Mat img2, img3;

    img2 = image.clone();

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cv::Vec3b p = img2.at<cv::Vec3b>(y,x);
        sprintf(text, "R=%d, G=%d, B=%d", p[2], p[1], p[0]);
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        // cv::cvtColor(image, img3, cv::BGR2HSV);
        cv::Vec3b p = img3.at<cv::Vec3b>(y,x);
        sprintf(text, "H=%d, S=%d, V=%d", p[0], p[1], p[2]);
    }
    else
        sprintf(text, "x=%d, y=%d", x, y);

    cv::putText(img2, text, cv::Point(5,50), cv::FONT_HERSHEY_PLAIN, 3.0, CV_RGB(0,0,255), 2.0);
    cv::imshow("Image", img2);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "minimal <tflite model> <labels file> <image path>\n");
        return 1;
    }
    const char* model_file = argv[1];
    const char* labels_file = argv[2];
    const char* image_file = argv[3];

    // Load class names
    std::vector<std::string> class_names;
    std::ifstream labels(labels_file);
    std::string line;
    while (std::getline(labels, line)) {
        class_names.push_back(line);
    }

    // Load the TFLite model from file
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_file);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Build the TFLite interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter" << std::endl;
        return 1;
    }

    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return 1;
    }

    // Get the input and output tensors
    int input = interpreter->inputs()[0];
    int output_boxes = interpreter->outputs()[0];
    int output_classes = interpreter->outputs()[1];
    int output_scores = interpreter->outputs()[2];
    int output_num_detections = interpreter->outputs()[3];

    //Check outputs
    int num_outputs = interpreter->outputs().size();
    std::cout << "Size of outputs: " + std::to_string(num_outputs) << std::endl;
    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* tensor = interpreter->output_tensor(i);
        std::cout << std::to_string(i) + " is " + std::to_string(interpreter->outputs()[i]) << std::endl;
        std::cout << "Output tensor " << i << ": " << tensor->name << std::endl;
        std::cout << "Output tensor " << i << ": " << GetTensorTypeString(tensor->type) << std::endl;
    }

    // Check Tensors  and Type
    for (int i = 0; i < interpreter->inputs().size(); ++i) {
        TfLiteTensor* tensor = interpreter->input_tensor(i);
        TfLiteIntArray* dims = tensor->dims;
        std::cout << "Input tensor " << i << " has shape ";
        for (int j = 0; j < dims->size; ++j) {
            std::cout << dims->data[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "Input tensor " << i << ": " << GetTensorTypeString(tensor->type) << std::endl;
    }

    for (int i = 0; i < interpreter->outputs().size(); ++i) {
        TfLiteTensor* tensor = interpreter->output_tensor(i);
        TfLiteIntArray* dims = tensor->dims;
        std::cout << "Output tensor " << i << " has shape ";
        for (int j = 0; j < dims->size; ++j) {
            std::cout << dims->data[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "Output tensor " << i << ": " << GetTensorTypeString(tensor->type) << std::endl;
    }

    cv::namedWindow("Image", 0);

    // Load the input image
    image = cv::imread(image_file);
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }
    

    // Resize the image to the input size of the model
    int64 start = cv::getTickCount();
    
    cv::Mat resized;
    auto input_tensor = interpreter->input_tensor(input);
    cv::Size size(input_tensor->dims->data[2],
                    input_tensor->dims->data[1]);
    cv::resize(image, resized, size);

    // Copy the image data to the input tensor
    // Here, interpreter->typed_input_tensor<uint8_t>(input) returns a pointer to the input tensor 
    // which will be used when invoking. Hence, on line 192, 
    // memcpy is basically just writing the input for inference
    uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(input);
    int input_height = size.height;
    int input_width = size.width;
    int input_channels = input_tensor->dims->data[3];
    cv::Mat input_mat(input_height, input_width, CV_8UC3);
    resized.convertTo(input_mat, CV_8UC3);

    memcpy(input_data, input_mat.data, input_height * input_width * input_channels * sizeof(uint8_t));

    // Run the model inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    // Get the output tensors
    // Here since, Tensor ID 598 which is OutputTensor0 is of type kTfLiteFloat32 it is float,
    // if it was kTfLiteUInt8, then it would have been uint8_t and so on
    // And same applies to all the other output tensors.
    // The description of output tensor will be found on their documentation pages or source code itself.
    // Or if it is your own custom model, you should have no problem defining these.
    // Please note that if you were using interpreter->typed_tensor instead of typed_output_tensor, 
    // you wil have to give the exact id for the output tensor which can be found using
    // interpreter->outputs()(i) where i is the ith output tensor.
    float* boxes = interpreter->typed_output_tensor<float>(0);
    if (boxes == nullptr) {
        std::cerr << "Failed to get the boxes." << std::endl;
        return -1;
    }

    float* classes = interpreter->typed_output_tensor<float>(1);
    if (classes == nullptr) {
        std::cerr << "Failed to get the classes." << std::endl;
        return -1;
    }

    float* scores = interpreter->typed_output_tensor<float>(2);
    if (scores == nullptr) {
        std::cerr << "Failed to get the scores." << std::endl;
        return -1;
    }
    
    float* num_detections = interpreter->typed_output_tensor<float>(3);
    if (num_detections == nullptr) {
        std::cerr << "Failed to get the number of detections." << std::endl;
        return -1;
    }

    // Draw the bounding boxes on the image
    const int DETECTIONS_MAX = 5;
    int n_det = 0;
    for (int i = 0; i < *num_detections; i++) {
        if(n_det >= DETECTIONS_MAX) break;      
        float score = static_cast<float>(scores[i]);
        if (score < 0.5) {
            continue;
        }
        std::cout << "Score: " + std::to_string(score) << std::endl;
        int class_id = (int)classes[i];
        float ymin = boxes[i * 4];
        float xmin = boxes[i * 4 + 1];
        float ymax = boxes[i * 4 + 2];
        float xmax = boxes[i * 4 + 3];

        float newXmin = xmin * image.cols;
        float newYmin = ymin * image.rows;
        float newXmax = (xmax - xmin) * image.cols;
        float newYmax = (ymax - ymin) * image.rows;
        std::cout << "BBOX: ( " + std::to_string(newXmin) + ", " + std::to_string(newYmin) + ", " + std::to_string(newXmax) + ", " + std::to_string(newYmax) + ")" << std::endl;
        cv::Rect box(newXmin, newYmin, newXmax, newYmax);
        cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, class_names[class_id], cv::Point(newXmin, newYmin - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(0, 0, 255), 2);
        n_det++;
    }
    int64 end = cv::getTickCount();
    double fps = cv::getTickFrequency() / (end - start);
    std::string fpsText = "FPS: " + std::to_string(fps);
    cv::putText(image, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                3, cv::Scalar(0, 0, 255), 2);
    cv::imshow("Image", image);
    cv::resizeWindow("Image", 500,500);
    cv::waitKey(0);
    return 0;
}