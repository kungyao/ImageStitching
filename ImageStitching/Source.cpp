#include <iostream>
#include <vector>
#include <string>
#include "FeatureUtil.h"

using namespace std;

void visual_feature(const std::vector<cv::Mat> &images, const FeatureList &featureList);

int main() {
	//string testDataPath = "./TestData/Chess_Board.png";
	string testDataPath = "./TestData/denny/denny00.jpg";
	std::vector<cv::Mat> images(0);
	images.push_back(cv::imread(testDataPath));
	testDataPath = "./TestData/denny/denny01.jpg";
	images.push_back(cv::imread(testDataPath));

	FeatureList tmp = Feature::HarrisCorner(images);
	const auto matches = FeatureMatch::Match(tmp);
	ImageUtil::GenerateMatchResult(images, tmp, matches);

	return 0;
}
