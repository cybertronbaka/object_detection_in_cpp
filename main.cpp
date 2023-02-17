#include <iostream>
#include <fstream>
#include <vector>
#include <optional>
#include <signal.h>


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

using namespace cv;
using namespace std;
using namespace tflite;

#define CAM_ID 0
#define CAM_API_ID CAP_ANY
#define DETECTIONS_MAX 5
#define RED Scalar(0, 0, 255)
#define LABEL_TEXT_POS Point(10, 20)


bool camInitiated = false;
VideoCapture cap;

void my_handler(int signum){
    if(!camInitiated) {
	return;
    }
    if(!cap.isOpened()){
	return;
    } else {
	camInitiated = false;
	cap.release();
	cout << "VideoCapture Released" << endl;
    }

    cout << "Quiting program..." << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "minimal <tflite model> <labels file>\n");
        return 1;
    }
    const char* model_file = argv[1];
    const char* labels_file = argv[2];

    // Initialize variables
   // VideoCapture cap;
    Mat image;
    // Load class names
    vector<string> class_names;
    ifstream labels(labels_file);
    string line;
    while (getline(labels, line)) {
        class_names.push_back(line);
    }

    // Load the TFLite model from file
    unique_ptr<FlatBufferModel> model =
        FlatBufferModel::BuildFromFile(model_file);
    if (!model) {
        cerr << "Failed to load model" << endl;
        return 1;
    }

    // Build the TFLite interpreter
    ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<Interpreter> interpreter;
    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        cerr << "Failed to build interpreter" << endl;
        return 1;
    }

    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "Failed to allocate tensors" << endl;
        return 1;
    }

    // Get the input and output tensors
    int input = interpreter->inputs()[0];

    cap.open(CAM_ID, CAM_API_ID);
    if(!cap.isOpened()){
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    camInitiated = true;
    cout << "Start grabbing" << endl;
    //namedWindow("Image", 0);
    //resizeWindow("Image", 500,500);


    Mat resized;
    auto input_tensor = interpreter->input_tensor(0);
    Size size(input_tensor->dims->data[2], input_tensor->dims->data[1]);
    int input_height = size.height;
    int input_width = size.width;
    int input_channels = input_tensor->dims->data[3];
    Mat input_mat(input_height, input_width, CV_8UC3);
    int n_det = 0;

    int64 start;

    signal (SIGINT,my_handler);

    for (;;){
	if(!camInitiated){
	    break;
	}
        start = getTickCount();
        cap.read(image);
        if(image.empty()){
            cerr << "ERROR! blank frame grabbed\n";
            return 1;
        }

        // Resize the image to the input size of the model
        resize(image, resized, size);

        // Copy the image data to the input tensor
        uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(0);

        resized.convertTo(input_mat, CV_8UC3);

        memcpy(input_data, input_mat.data, input_height * input_width * input_channels * sizeof(uint8_t));

        // Run the model inference
        if (interpreter->Invoke() != kTfLiteOk) {
            cerr << "Failed to invoke interpreter" << endl;
            return 1;
        }

        // Get the output tensors
        float* boxes = interpreter->typed_output_tensor<float>(0);
        // if (boxes == nullptr) {
        //     cerr << "Failed to get the boxes." << endl;
        //     return -1;
        // }
        float* classes = interpreter->typed_output_tensor<float>(1);
        // if (classes == nullptr) {
        //     cerr << "Failed to get the classes." << endl;
        //     return -1;
        // }
        float* scores = interpreter->typed_output_tensor<float>(2);
        // if (scores == nullptr) {
        //     cerr << "Failed to get the scores." << endl;
        //     return -1;
        // }
        float* num_detections = interpreter->typed_output_tensor<float>(3);
        // if (num_detections == nullptr) {
        //     cerr << "Failed to get the number of detections." << endl;
        //     return -1;
        // }

        // Draw the bounding boxes on the image
        n_det = 0;
        for (int i = 0; i < *num_detections; i++) {
            if(n_det >= DETECTIONS_MAX) break;      

            float score = static_cast<float>(scores[i]);

            if (score < 0.5) continue;

            int class_id = (int)classes[i];
            float ymin = boxes[i * 4] * image.rows;
            float xmin = boxes[i * 4 + 1] * image.cols;
            float ymax = boxes[i * 4 + 2] * image.rows;
            float xmax = boxes[i * 4 + 3] * image.cols;

            rectangle(image, Rect(xmin, ymin, xmax - xmin, ymax - ymin), RED, 2);
	    cout << "DETECTED: " << class_names[class_id] << endl;
            putText(image, class_names[class_id], Point(xmin, ymin - 10), FONT_HERSHEY_SIMPLEX, 0.5,RED, 2);
            n_det++;
        }
        double fps = getTickFrequency() / (getTickCount() - start);
        putText(image, "FPS: " + to_string(fps), LABEL_TEXT_POS, FONT_HERSHEY_SIMPLEX, 0.5, RED, 2);
	cout << "FPS: " + to_string(fps) << endl;
        //imshow("Image", image);
        //if(waitKey(1) >= 0)
        //    break;
    }

    return 0;
}
