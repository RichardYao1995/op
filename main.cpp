#include <iostream>
#include <chrono>
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "rtabmap/core/util2d.h"

using namespace std;
using namespace cv;

Eigen::Vector3d projectDisparityTo3D(const cv::Point2f & pt, float disparity)
{
    if(disparity > 0.0f)
    {
        float W = 0.35 / disparity;
        return Eigen::Vector3d(762.72 * W, -(pt.x - 640) * W, -(pt.y - 360) * W);
    }
    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    return Eigen::Vector3d(bad_point, bad_point, bad_point);
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const cv::Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }

}

Eigen::Matrix3d getRFromrpy(const Eigen::Vector3d& rpy)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d ea(rpy(0),rpy(1),rpy(2));
    R = Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitX());
    return R;
}

int main ()
{
    Eigen::Matrix<double,1,6> pose1, pose2;
    pose1 << 15.244168, 3.984536, -0.000523, 0.000792, 0.000351, -1.216275;
    pose2 << 15.544473, 4.096562, 0.000407, -0.000432, 0.000115, -1.208648;
    Eigen::Matrix3d R1 = getRFromrpy(Eigen::Vector3d(pose1(3), pose1(4), pose1(5)));
    Eigen::Matrix4d T1;
    Eigen::Vector3d t1;t1 << pose1(0), pose1(1), pose1(2);
    t1 = R1 * (-t1);
    T1 << R1(0, 0), R1(0, 1), R1(0, 2), pose1(0),
          R1(1, 0), R1(1, 1), R1(1, 2), pose1(1),
          R1(2, 0), R1(2, 1), R1(2, 2), pose1(2),
          0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d R2 = getRFromrpy(Eigen::Vector3d(pose2(3), pose2(4), pose2(5)));
    Eigen::Matrix4d T2;
    Eigen::Vector3d t2;t2 << pose2(0), pose2(1), pose2(2);
    t2 = R2 * (-t2);
    T2 << R2(0, 0), R2(0, 1), R2(0, 2), pose2(0),
          R2(1, 0), R2(1, 1), R2(1, 2), pose2(1),
          R2(2, 0), R2(2, 1), R2(2, 2), pose2(2),
          0.0, 0.0, 0.0, 1.0;
    cv::Mat first;
    cv::Mat color = cv::imread("/home/uisee/Data/stereo-0/left/0000000138.tiff");
    uint row = color.rows;
    uint col = color.cols;
    cv::cvtColor(cv::imread("/home/uisee/Data/stereo-0/left/0000000138.tiff"), first, CV_BGR2GRAY);
    cv::Mat first_right;
    cv::cvtColor(cv::imread("/home/uisee/Data/stereo-0/right/0000000138.tiff"), first_right, CV_BGR2GRAY);
    cv::Mat second;
    cv::cvtColor(cv::imread("/home/uisee/Data/stereo-0/left/0000000139.tiff"), second, CV_BGR2GRAY);
    cv::Mat second_right;
    cv::cvtColor(cv::imread("/home/uisee/Data/stereo-0/right/0000000139.tiff"), second_right, CV_BGR2GRAY);
    cv::Mat dest1 = rtabmap::util2d::disparityFromStereoImages(first, first_right);
    dest1.convertTo(dest1,CV_32FC1,1.0/16);
    cv::Mat dest2 = rtabmap::util2d::disparityFromStereoImages(second, second_right);
    dest2.convertTo(dest2,CV_32FC1,1.0/16);

    cv::Mat flow;
    auto start = std::chrono::system_clock::now();
    cv::calcOpticalFlowFarneback(first, second, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The optical flow calculation costs " << double(duration.count())
                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << " seconds" << std::endl;
    drawOptFlowMap(flow, color, 16, 1.5, Scalar(0, 255, 0));
    cv::imwrite("result.png", color);
    //cout << dest1.at<float>(500, 640);

    Mat depth_first_cov(row, col, CV_32FC1);
    Mat depth_first(row, col, CV_32FC1);
    for(int x = 0;x < row;x++)
    {
        for(int y = 0;y < col;y++)
        {
            if(dest1.at<float>(x, y) < 3.0)
                continue;
            double depth = 762.72 * 0.35 / dest1.at<float>(x, y);
            double sigma = depth - 762.72 * 0.35 / (dest1.at<float>(x, y) - 1);
            double sigma2 = sigma * sigma;
            depth_first_cov.ptr<float>(x)[y] = sigma2;
            depth_first.ptr<float>(x)[y] = depth;
        }
    }
    start = std::chrono::system_clock::now();
    Mat depth_second_cov(row, col, CV_32FC1);
    Mat depth_second(row, col, CV_32FC1);
    for(int x = 0;x < row;x++)
    {
        for(int y = 0;y < col;y++)
        {
            if(dest2.at<float>(x, y) < 3.0)
                continue;
            double depth = 762.72 * 0.35 / dest2.at<float>(x, y);
            double sigma = depth - 762.72 * 0.35 / (dest2.at<float>(x, y) - 1);
            double sigma2 = sigma * sigma;
            depth_second_cov.ptr<float>(x)[y] = sigma2;
            depth_second.ptr<float>(x)[y] = depth;
        }
    }
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The depth calculation costs " << double(duration.count())
                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << " seconds" << std::endl;

    start = std::chrono::system_clock::now();
    Eigen::Matrix3d R12 = R2 * R1.inverse();
    Eigen::Vector3d t12 = t2 - R12 * t1;
    for(uint x = 20;x < row - 20;x++)
    {
        for(uint y = 20;y < col - 20;y++)
        {
            if(dest1.at<float>(x, y) < 3.0)
                continue;
            Point2f& fxy = flow.at<Point2f>(x, y);
            double depth2 = depth_second.at<float>(x + fxy.y, y + fxy.x);
            //double d_cov = depth2 - 762.72 * 0.35 / (dest2.at<float>(500 + fxy.y, 640 + fxy.x) - 1);
            double sigma_second = depth_second_cov.at<float>(x + fxy.y, y + fxy.x);

            //double depth1 = depth_first.ptr<float>(x)[y];
            //double sigma = depth1 - 762.72 * 0.35 / (dest1.at<float>(500, 640) - 1);
            double sigma_first = depth_first_cov.ptr<float>(x)[y];

            cv::Point2f pt1(y, x);
            Eigen::Vector3d point1 = projectDisparityTo3D(pt1, dest1.at<float>(x, y));
            Eigen::Vector3d point12 = R12 * point1 + t12;
            //cout << point1 << point12;
            double depth12 = point12[0];
            double depth_fuse = (sigma_second * depth12 + sigma_first * depth2) / (sigma_second + sigma_first);
            double sigma_fuse = (sigma_second * sigma_first) / (sigma_second + sigma_first);

            depth_second_cov.at<float>(x + fxy.y, y + fxy.x) = sigma_fuse;
            depth_second.at<float>(x + fxy.y, y + fxy.x) = depth_fuse;
        }
    }
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The update costs " << double(duration.count())
                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << " seconds" << std::endl;

    return 0;
}
