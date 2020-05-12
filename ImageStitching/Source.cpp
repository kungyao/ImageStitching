﻿#include <iostream>
#include <vector>
#include <string>
#include "FeatureUtil.h"
#include "Common.h"

using namespace std;

int main() {
	//string testDataPath = "./TestData/Chess_Board.png";
	//string testDataPath = "./TestData/denny/denny05.jpg";
	//std::vector<cv::Mat> images(0);
	//images.push_back(cv::imread(testDataPath));
	//testDataPath = "./TestData/denny/denny06.jpg";
	//images.push_back(cv::imread(testDataPath));

	std::vector<cv::Mat> images = ImageUtil::LoadImageList("./TestData/denny/");

	FeatureList featureList = Feature::HarrisCorner(images);
	const auto matches = FeatureMatch::Match(featureList);
	ImageUtil::GenerateMatchResult(images, featureList, matches);
	const auto alignments = ImageMatcher::Match(images, featureList, matches);
	cv::Mat panorama = ImageBlender::Blend(images, alignments, true);
	ImageUtil::WriteImage(panorama, "panorama");
	return 0;
}
