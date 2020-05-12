#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

const std::string outPath = "./Feature/";

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

	friend Vec2x operator +(const Vec2x& lhs, const Vec2x &rhs) {
		return Vec2x(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	friend Vec2x operator -(const Vec2x& lhs, const Vec2x& rhs) {
		return Vec2x(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	Vec2x& operator +=(const Vec2x& rhs) {
		x += rhs.x;
		y += rhs.y;
		return *this;
	}

	friend Vec2x operator *(const Vec2x& lhs, const T& val) {
		return Vec2x(lhs.x * val, lhs.y * val);
	}

	friend std::ostream& operator<<(std::ostream& os, const Vec2x& vec) {
		std::cout << "(x : " << vec.x << ", y : " << vec.y << ")";
		return os;
	}
};

#define Vec2i Vec2x<int>
#define Vec2f Vec2x<float>

class ResponseInfo {
public:
	float r;
	float c;
	Vec2i pos;
	ResponseInfo(float _r, float _c, Vec2i v2i) {
		r = _r;
		c = _c;
		pos = v2i;
	}
};

class FeatureInfo {
public:
	// feature position
	std::vector<Vec2i> pos;
	// description
	std::vector<std::vector<int>> descs;
	FeatureInfo() {
		pos.clear();
		descs.clear();
	}
	FeatureInfo(const std::vector<ResponseInfo>& tmpFeature, const int& maxSizeOfFeature) {
		pos.reserve(maxSizeOfFeature);
		for (int i = 0; i < tmpFeature.size() && i < maxSizeOfFeature; i++)pos.push_back(tmpFeature[i].pos);
		pos.resize(pos.size());
	}
};

class ImageUtil {
public:
	static void GenerateMatchResult(const std::vector<cv::Mat>& images, const std::vector<FeatureInfo>& fInfos, const std::vector<std::vector<Vec2i>>& matches) {
		for (int i = 0; i < matches.size(); i++) {
			const auto& lhsFeaturePoints = fInfos[i].pos;
			const auto& rhsFeaturePoints = fInfos[i + 1].pos;

			std::vector<cv::KeyPoint> lhsKp;
			std::vector<cv::KeyPoint> rhsKp;
			std::vector<cv::DMatch> dms;

			for (int j = 0; j < lhsFeaturePoints.size(); j++)
				lhsKp.push_back(cv::KeyPoint(cv::Point(lhsFeaturePoints[j].y, lhsFeaturePoints[j].x), 2));
			for (int j = 0; j < rhsFeaturePoints.size(); j++)
				rhsKp.push_back(cv::KeyPoint(cv::Point(rhsFeaturePoints[j].y, rhsFeaturePoints[j].x), 2));

			for (int j = 0; j < matches[i].size(); j++)
				dms.push_back(cv::DMatch(matches[i][j].x, matches[i][j].y, -1));
			//#ifdef WRITE_MIDDLE_STEP_IMAGE
			cv::Mat out;
			cv::drawMatches(images[i], lhsKp, images[i + 1], rhsKp, dms, out, cv::Scalar(0, 0, 255));
			cv::imwrite(outPath + "match_" + std::to_string(i) + "_" + std::to_string(i + 1) + ".png", out);
		}
	}
	static std::vector<cv::Mat> LoadImageList(const std::string& dir) {
		std::vector<cv::Mat> img_list(0);
		const std::string image_list_dir = "pano.txt";
		std::fstream in;
		in.open(dir + image_list_dir);

		if (in) {
			std::string imageName;
			// float focalLength = 0.0f;
			while (in >> imageName)
			{
				std::string imagePath = dir + imageName;
				cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
				img_list.push_back(image.clone());
				// focalLengths.push_back(focalLength);
			}
		}
		else {
			throw(dir + " doesn't have pano.txt including the image names");
		}
		return img_list;
	}
	static void WriteImage(const cv::Mat& image, const std::string& name) {
		cv::imwrite(outPath + name + ".png", image);
	}
};