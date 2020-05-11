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

#define WRITE_MIDDLE_STEP_IMAGE True

template <typename T>
class Vec2x {
public:
	T x = 0;
	T y = 0;

	Vec2x() {
		x = 0;
		y = 0;
	}
	Vec2x(T _x, T _y) {
		x = _x;
		y = _y;
	}

	Vec2x abs() {
		return Vec2x(std::abs(x), std::abs(y));
	}

	//Vec2x operator +(const Vec2x &rhs) {
	//	return Vec2x(x * rhs.x, y * rhs.y);
	//}

	Vec2x &operator +=(const Vec2x &rhs) {
		x += rhs.x;
		y += rhs.y;
		return *this;
	}

	friend Vec2x operator *(const Vec2x &lhs, const T &val) {
		return Vec2x(lhs.x * val, lhs.y * val);
	}

	friend ostream& operator<<(ostream& os, const Vec2x& vec) {
		cout << "(x : " << vec.x << ", y : " << vec.y << ")";
		return os;
	}
};

#define Vec2i Vec2x<int>
#define Vec2f Vec2x<float>

using feature_pair = std::pair<float, Vec2i>;

class FeatureInfo {
public:
	// feature position
	std::vector<Vec2i> pos;
	// description
	vector<vector<int>> descs;
	FeatureInfo() {
		pos.clear();
		descs.clear();
	}
	FeatureInfo(const std::vector<feature_pair> &tmpFeature, const int &maxSizeOfFeature) {
		pos.reserve(maxSizeOfFeature);
		for (int i = 0; i < tmpFeature.size() && i < maxSizeOfFeature; i++)pos.push_back(tmpFeature[i].second);
		pos.resize(pos.size());
	}
};
#define FeatureList std::vector<FeatureInfo>

class Feature {
private:
	static int maxFeatureSize;
	// sift
	static void featureDesctiptor(FeatureInfo &fInfo, const cv::Mat &angle) {
		const auto& pos = fInfo.pos;
		auto& descs = fInfo.descs;
		for (int i = 0; i < pos.size(); i++) {
			int x = pos[i].x;
			int y = pos[i].y;

			vector<int> votes(128, 0);
			int blockStart[4] = { -8, -4, 1, 5 };
			for (int by = 0; by < 4; by++)
			{
				int y_ = y + blockStart[by];
				for (int bx = 0; bx < 4; bx++)
				{
					int x_ = x + blockStart[bx];
					for (int dy = 0; dy < 4; dy++)
					{
						for (int dx = 0; dx < 4; dx++)
						{
							int idx = 8 * (4 * by + bx) + floor(angle.at<float>(y_ + dy, x_ + dx));
							votes[idx]++;
						}
					}
				}
			}
			descs.push_back(votes);
		}
	}
public:
	static FeatureList HarrisCorner(const std::vector<cv::Mat> &images, float threshold = 10000) {
		FeatureList featurePoints(0);
		featurePoints.reserve(images.size());
		float k = 0.05;
		for (const cv::Mat &img : images) {
			cv::Mat grayImg;
			cv::cvtColor(img, grayImg, CV_BGR2GRAY);
			grayImg.convertTo(grayImg, CV_32FC1);

			Mat Ix, Iy;
			Mat kernelX(1, 3, CV_32F);
			kernelX.at<float>(0, 0) = -1.0f;
			kernelX.at<float>(0, 1) = 0.0f;
			kernelX.at<float>(0, 2) = 1.0f;
			filter2D(grayImg, Ix, CV_32F, kernelX);

			Mat kernelY(3, 1, CV_32F);
			kernelY.at<float>(0, 0) = -1.0f;
			kernelY.at<float>(1, 0) = 0.0f;
			kernelY.at<float>(2, 0) = 1.0f;
			filter2D(grayImg, Iy, CV_32F, kernelY);

#ifdef WRITE_MIDDLE_STEP_IMAGE
			cv::imwrite("./Feature/grad_x_img.png", Ix);
			cv::imwrite("./Feature/grad_y_img.png", Iy);
#endif // WRITE_MIDDLE_STEP_IMAGE

			cv::Mat gaus_Ix2;
			cv::Mat gaus_Iy2;
			cv::Mat gaus_Ixy;
			cv::GaussianBlur(Ix.mul(Ix), gaus_Ix2, cv::Size(5, 5), 3);
			cv::GaussianBlur(Iy.mul(Iy), gaus_Iy2, cv::Size(5, 5), 3);
			cv::GaussianBlur(Ix.mul(Iy), gaus_Ixy, cv::Size(5, 5), 3);

			cv::Mat trace = gaus_Ix2 + gaus_Iy2;
			cv::Mat R = (gaus_Ix2.mul(gaus_Iy2) - gaus_Ixy.mul(gaus_Ixy)) - k * trace.mul(trace);

			int dx_step[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
			int dy_step[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };
			auto isLocalMaximum = [&](const cv::Mat img_r, const int &x, const int &y) {
				for (const int& dx : dx_step) {
					int rx = x + dx;
					if (rx < 0 || rx >= img_r.cols)
						continue;
					for (const int& dy : dy_step) {
						int ry = y + dy;
						if (ry < 0 || ry >= img_r.rows)
							continue;
						if (img_r.at<float>(ry, rx) > img_r.at<float>(y, x))
							return false;
					}
				}
				return true;
			};

			std::vector<feature_pair> tmpFeature;
			//printf("rows : %d, cols : %d\n", img.rows, img.cols);
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					float r = R.at<float>(i, j);
					if (r > threshold && isLocalMaximum(R, j, i)) {
						tmpFeature.push_back(feature_pair(r, Vec2i(i, j)));
					}
				}
			}

			auto r_val_compare = [&](const feature_pair &f1, const feature_pair& f2) {
				return f1.first > f2.first;
			};
			std::sort(tmpFeature.begin(), tmpFeature.end(), r_val_compare);
			FeatureInfo fea(tmpFeature, maxFeatureSize);

			cv::Mat angle(Ix.size(), CV_32F);
			for (int i = 0; i < Ix.rows; i++)
				for (int j = 0; j < Ix.cols; j++)
					angle.at<float>(i, j) = cv::fastAtan2(Ix.at<float>(i, j), Iy.at<float>(i, j)) / 45.0f;
			featureDesctiptor(fea, angle);

			featurePoints.push_back(fea);
		}
		return featurePoints;
	}
};

int Feature::maxFeatureSize = 500;