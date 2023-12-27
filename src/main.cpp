#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"


int main(int argc, char* argv[])
{
    const float confThreshold = 0.8f;
    const float iouThreshold = 0.8f;


    bool isGPU = true;
    const std::string classNamesPath = "/date2/wangzijian/yolov5-onnxruntime/models/coco.names";
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imagePath = "/date2/wangzijian/yolov5-onnxruntime/images/bus.jpg";
    const std::string modelPath = "/date2/wangzijian/yolov5-onnxruntime/models/best.onnx";

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    YOLODetector detector {nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    try
    {
        detector = YOLODetector(modelPath, isGPU, cv::Size(320, 320));
        std::cout << "Model was initialized." << std::endl;

        image = cv::imread(imagePath);
        result = detector.detect(image, confThreshold, iouThreshold);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    utils::visualizeDetection(image, result, classNames);


    // cv::imshow("result", image);
    cv::imwrite("/date2/wangzijian/yolov5-onnxruntime/images/result.jpg", image);
    // cv::waitKey(0);

    return 0;
}
