#include <iostream>
#include <vector>
#include <string>
#include "FeatureUtil.h"

using namespace std;

void visual_feature(const std::vector<cv::Mat> &images, const FeatureList &featureList);

int main() {
	//test_weight_list();
	//test_gradient_map();
	
	//string testDataPath = "./TestData/Chess_Board.png";
	string testDataPath = "./TestData/denny/denny00.jpg";
	std::vector<cv::Mat> images(0);
	images.push_back(cv::imread(testDataPath));
	FeatureList tmp = Feature::HarrisCorner(images);
	visual_feature(images, tmp);
	return 0;
}

void visual_feature(const std::vector<cv::Mat> &images, const FeatureList &featureList) {
	int half_x = 0, half_y = 0, WINDOW_SIZE = 11;
	half_x = half_y = (WINDOW_SIZE - 1) / 2;
	for (int i = 0; i < images.size(); i++) {
		const cv::Mat &img = images[i];
		const auto &fPoint = featureList[i].pos;
		cv::Mat tmpImg;
		img.copyTo(tmpImg);
		for (const auto &p : fPoint) {
			//printf("%d %d %d %d\n", p.y - half_y, p.x - half_x, p.y + half_y, p.x + half_x);
			cv::rectangle(tmpImg, cv::Rect(p.y - half_y, p.x - half_x, WINDOW_SIZE, WINDOW_SIZE), Scalar(0, 0, 255));
		}
		string tmpOut = "./Feature/" + to_string(i);
		cv::imwrite(tmpOut + "_image.png", tmpImg);
	}
}