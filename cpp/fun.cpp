#include <iostream>
#include <limits>
#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat computeEnergyMatrix(const cv::Mat& _image)
{
    cv::Mat sobelX, sobelY;
    cv::Sobel(_image, sobelX, CV_32F, 1, 0);
    cv::Sobel(_image, sobelY, CV_32F, 0, 1);
    cv::Mat energyMatrix = cv::abs(sobelX) + cv::abs(sobelY);
    cv::transform(energyMatrix, energyMatrix, cv::Matx13f(1,1,1));
    return energyMatrix;
}

#define MAX_DEVIATION   2
std::vector<int> findVerticalSeam(const cv::Mat& _energyMatrix)
{
    int m = _energyMatrix.rows, n = _energyMatrix.cols;

    cv::Mat pathEnergy = cv::Mat::zeros(cv::Size(n, m), CV_32FC1) + std::numeric_limits<float>::max();
    _energyMatrix(cv::Rect(0, 0, n, 1)).copyTo(pathEnergy(cv::Rect(0, 0, n, 1)));
    cv::Mat offsets = cv::Mat::zeros(cv::Size(n, m), CV_32SC1);
    for(int i = 1; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int o = -1; o <= 1; ++o) {
                if(j + o >= 0 && j + o < n) {
                    float offsetCost = pathEnergy.at<float>(i - 1, j + o) + _energyMatrix.at<float>(i, j);
                    if(pathEnergy.at<float>(i, j) > offsetCost) {
                        pathEnergy.at<float>(i, j) = offsetCost;
                        offsets.at<int>(i, j) = o;
                    }
                }
            }
        }
    }
    std::vector<int> seam(m);
    seam[m - 1] = 0;
    for(int i = 1; i < n; ++i)
        if(pathEnergy.at<float>(m - 1, i) < pathEnergy.at<float>(m - 1, seam[m - 1]))
            seam[m - 1] = i;
    for(int i = m - 1; i > 0; --i)
        seam[i - 1] = seam[i] + offsets.at<int>(i, seam[i]);
    return seam;
}

cv::Mat removeVerticalSeam(const cv::Mat& _image, const std::vector<int>& _seam)
{
    int m = _image.rows, n = _image.cols;
    cv::Mat result(cv::Size(n - 1, m), CV_8UC3);
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < _seam[i]; ++j) 
            result.at<cv::Vec3b>(i, j) = _image.at<cv::Vec3b>(i, j);
        for(int j = _seam[i]; j < n - 1; ++j)
            result.at<cv::Vec3b>(i, j) = _image.at<cv::Vec3b>(i, j + 1);
    }
    return result;
}

int main() {

    auto image = cv::imread("../penguines.jpg");

    for(int i = 0; i < 250; ++i) {
        auto energyMatrix = computeEnergyMatrix(image);
        auto seam = findVerticalSeam(energyMatrix);
        image = removeVerticalSeam(image, seam);
        std::cout << (i + 1) << " seam(s) removed." << std::endl;
    }

    cv::imwrite("../seam_carved.jpg", image);

    return 0;
}