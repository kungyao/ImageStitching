#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
//#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

template <typename T>
class Vec2x {
public:
	T x = 0;
	T y = 0;

	friend ostream& operator<<(ostream& os, const Vec2x& vec) {
		cout << "(x : " << vec.x << ", y : " << vec.y << ")";
		return os;
	}
};

#define Vec2i Vec2x<int>
#define Vec2f Vec2x<float>

class Feature {
public:
	// w(x, y) = exp(-(x^2 + y^2) / (2 * variance))
	static std::vector<std::vector<float>> genWeightList(int window_size) {
		std::vector<std::vector<float>> weights(0);
		weights.reserve(window_size);
		int ori_x, ori_y;
		ori_x = ori_y = (window_size - 1) / 2;
		float suqareSumOfWeight = 0;
		for (int i = 0; i < window_size; i++) {
			std::vector<float> tpmw(0);
			tpmw.reserve(window_size);
			int offset_y = i - ori_y;
			for (int j = 0; j < window_size; j++) {
				int offset_x = j - ori_x;
				float diff_to_ori = offset_y * offset_y + offset_x * offset_x;
				suqareSumOfWeight += diff_to_ori;
				tpmw.push_back(diff_to_ori);
			}
			weights.push_back(tpmw);
		}

		int filterSize = window_size * window_size;
		float mean = suqareSumOfWeight / filterSize;
		float variance = suqareSumOfWeight / filterSize - mean * mean;
		variance *= 2;

		for (int i = 0; i < window_size; i++) {
			for (int j = 0; j < window_size; j++) {
				float *w = &weights[i][j];
				*w = std::exp(*w / variance);
			}
		}

		return weights;
	}

	// x和y方向
	static std::vector<std::vector<Vec2f>> genImageGradient(const cv::Mat &img) {
		cv::Mat grayImg;
		img.copyTo(grayImg);
		// cvtColor : src會被釋放?
		cv::cvtColor(grayImg, grayImg, CV_BGR2GRAY);
		std::vector<std::vector<Vec2f>> gradientMap(0);
		gradientMap.reserve(grayImg.rows);
		// rows : height
		// cols : width
		for (int i = 0; i < grayImg.rows; i++) {
			std::vector<Vec2f> tmp;
			tmp.reserve(grayImg.cols);
			for (int j = 0; j < grayImg.cols; j++) {
				Vec2f g;
				// x direction
				if (j - 1 >= 0)g.x -= grayImg.at<uchar>(i, j - 1);
				if (j + 1 < grayImg.cols)g.x += grayImg.at<uchar>(i, j + 1);
				// y direction
				if (i - 1 >= 0)g.y -= grayImg.at<uchar>(i - 1, j);
				if (i + 1 < grayImg.rows)g.y += grayImg.at<uchar>(i + 1, j);
				tmp.push_back(g);
			}
			gradientMap.push_back(tmp);
		}
		return gradientMap;
	}
public:
	static std::vector<std::vector<Vec2i>> HarrisCorner(const std::vector<cv::Mat> &images, int window_size) {
		std::vector<std::vector<float>> weightList = genWeightList(window_size);
		std::vector<std::vector<Vec2i>> featurePoints(0);
		featurePoints.reserve(images.size());
		for (const cv::Mat &img : images) {
			std::vector<Vec2i> featureOfimg(0);
			featureOfimg.reserve(128);

			featureOfimg.resize(featureOfimg.size());
			featurePoints.push_back(featureOfimg);
		}
		return featurePoints;
	}
};
