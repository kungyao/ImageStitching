#include <iostream>
#include <vector>
#include <string>
#include "FeatureUtil.h"

#define MyVec(x) Vec3b(x,x,x) 

using namespace std;

void test_weight_list();
void test_gradient_map();
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
	int half_x = 0, half_y = 0;
	half_x = half_y = (WINDOW_SIZE - 1) / 2;
	for (int i = 0; i < images.size(); i++) {
		const cv::Mat &img = images[i];
		const auto &fPoint = featureList[i];
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

void test_weight_list() {
	auto list = Gaussian::genGaussWeight(WINDOW_SIZE);
	for (auto l1 : list) {
		for (auto l2 : l1) {
			cout << l2 << " ";
		}
		cout << "\n";
	}
	/*	
		0.907286 0.926804 0.941001 0.949624 0.952516 0.949624 0.941001 0.926804 0.907286
		0.926804 0.946741 0.961244 0.970052 0.973006 0.970052 0.961244 0.946741 0.926804
		0.941001 0.961244 0.975969 0.984912 0.987912 0.984912 0.975969 0.961244 0.941001
		0.949624 0.970052 0.984912 0.993937 0.996964 0.993937 0.984912 0.970052 0.949624
		0.952516 0.973006 0.987912 0.996964 1 0.996964 0.987912 0.973006 0.952516
		0.949624 0.970052 0.984912 0.993937 0.996964 0.993937 0.984912 0.970052 0.949624
		0.941001 0.961244 0.975969 0.984912 0.987912 0.984912 0.975969 0.961244 0.941001
		0.926804 0.946741 0.961244 0.970052 0.973006 0.970052 0.961244 0.946741 0.926804
		0.907286 0.926804 0.941001 0.949624 0.952516 0.949624 0.941001 0.926804 0.907286
	*/
}

void test_gradient_map() {
	Mat fakeImage(3, 5, CV_8UC3);
	fakeImage.at<Vec3b>(0, 0) = MyVec(0);
	fakeImage.at<Vec3b>(0, 1) = MyVec(20);
	fakeImage.at<Vec3b>(0, 2) = MyVec(50);
	fakeImage.at<Vec3b>(0, 3) = MyVec(10);
	fakeImage.at<Vec3b>(0, 4) = MyVec(0);
	fakeImage.at<Vec3b>(1, 0) = MyVec(30);
	fakeImage.at<Vec3b>(1, 1) = MyVec(200);
	fakeImage.at<Vec3b>(1, 2) = MyVec(70);
	fakeImage.at<Vec3b>(1, 3) = MyVec(45);
	fakeImage.at<Vec3b>(1, 4) = MyVec(10);
	fakeImage.at<Vec3b>(2, 0) = MyVec(7);
	fakeImage.at<Vec3b>(2, 1) = MyVec(15);
	fakeImage.at<Vec3b>(2, 2) = MyVec(55);
	fakeImage.at<Vec3b>(2, 3) = MyVec(30);
	fakeImage.at<Vec3b>(2, 4) = MyVec(0);
	auto list2 = GradientImage(fakeImage, NULL);
	for (auto l1 : list2.gradient) {
		for (auto l2 : l1) {
			cout << l2 << " ";
		}
		cout << "\n";
	}
	/*
		(x : 2, y : 3) (x : 6, y : 23) (x : -1, y : 8) (x : -6, y : 5) (x : -1, y : 1)
		(x : 23, y : 29) (x : 5, y : 117) (x : -18, y : -4) (x : -7, y : 117) (x : -5, y : 134)
		(x : 119, y : -3) (x : -27, y : 67) (x : -1, y : 23) (x : 132, y : 43) (x : -118, y : 119)
		(x : 90, y : -27) (x : 31, y : -1) (x : -42, y : 132) (x : 89, y : -118) (x : -48, y : -44)
		(x : 118, y : 0) (x : 132, y : -90) (x : -118, y : -31) (x : -44, y : -48) (x : 0, y : -120)
	*/
}