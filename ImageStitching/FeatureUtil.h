#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
//#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Common.h"

#define FeatureList std::vector<FeatureInfo>

//#define WRITE_MIDDLE_STEP_IMAGE

class Feature {
public:
	static int maxFeatureSize;
private:
	// sift
	static void featureDesctiptor(FeatureInfo &fInfo, const cv::Mat &angle) {
		const auto& pos = fInfo.pos;
		auto& descs = fInfo.descs;
		for (int i = 0; i < pos.size(); i++) {
			int y = pos[i].x;
			int x = pos[i].y;

			std::vector<int> votes(128, 0);
			int blockStart[4] = { -8, -4, 1, 5 };
			for (int by = 0; by < 4; by++)
			{
				int y_ = y + blockStart[by];
				for (int bx = 0; bx < 4; bx++)
				{
					int x_ = x + blockStart[bx];
					for (int dy = 0; dy < 4; dy++)
					{
						int tmp_y = y_ + dy;
						if (tmp_y < 0 || tmp_y >= angle.rows)
							continue;
						for (int dx = 0; dx < 4; dx++)
						{
							int tmp_x = x_ + dx;
							if (tmp_x < 0 || tmp_x >= angle.cols)
								continue;
							// std::cout << y_ + dy << "  " << x_ + dx << "\n";
							int idx = 8 * (4 * by + bx) + std::floor(angle.at<float>(y_ + dy, x_ + dx));
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

#ifdef WRITE_MIDDLE_STEP_IMAGE
		int i = 0;
#endif
		for (const cv::Mat &img : images) {
			cv::Mat grayImg;
			cv::cvtColor(img, grayImg, CV_BGR2GRAY);
			grayImg.convertTo(grayImg, CV_32FC1);

			// use sobel filter to calculate image gradient
			cv::Mat Ix, Iy;
			cv::Mat kernelX(3, 3, CV_32F);
			kernelX.at<float>(0, 0) = 1.0f;
			kernelX.at<float>(0, 1) = 0.0f;
			kernelX.at<float>(0, 2) = -1.0f;
			//
			kernelX.at<float>(1, 0) = 2.0f;
			kernelX.at<float>(1, 1) = 0.0f;
			kernelX.at<float>(1, 2) = -2.0f;
			//
			kernelX.at<float>(2, 0) = 1.0f;
			kernelX.at<float>(2, 1) = 0.0f;
			kernelX.at<float>(2, 2) = -1.0f;
			cv::filter2D(grayImg, Ix, CV_32F, kernelX);

			cv::Mat kernelY(3, 3, CV_32F);
			kernelY.at<float>(0, 0) = 1.0f;
			kernelY.at<float>(1, 0) = 0.0f;
			kernelY.at<float>(2, 0) = -1.0f;
			//
			kernelY.at<float>(0, 1) = 2.0f;
			kernelY.at<float>(1, 1) = 0.0f;
			kernelY.at<float>(2, 1) = -2.0f;
			//
			kernelY.at<float>(0, 2) = 1.0f;
			kernelY.at<float>(1, 2) = 0.0f;
			kernelY.at<float>(2, 2) = -1.0f;
			cv::filter2D(grayImg, Iy, CV_32F, kernelY);

#ifdef WRITE_MIDDLE_STEP_IMAGE
			cv::imwrite(outPath + "grad_x_img_" + std::to_string(i) + ".png", Ix);
			cv::imwrite(outPath + "grad_y_img_" + std::to_string(i) + ".png", Iy);
			i++;
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

			// https://cmsc426.github.io/pano/?fbclid=IwAR2pleqRn54yQgRzetwXA9pI2p4Hc-WjXbaB7wX9YItvjxKVAeLD3eOerdE#anms
			std::vector<ResponseInfo> tmpFeature;
			//printf("rows : %d, cols : %d\n", img.rows, img.cols);
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					float r = R.at<float>(i, j);
					if (r > threshold && isLocalMaximum(R, j, i)) {
						tmpFeature.push_back(ResponseInfo(std::numeric_limits<float>::max(), r, Vec2i(i, j)));
					}
				}
			}

			for (int i = 0; i < tmpFeature.size(); i++)
			{
				int ED = 1000000;
				for (int j = 0; j < tmpFeature.size(); j++)
				{
					if (tmpFeature[j].c > tmpFeature[i].c)
						ED = pow(tmpFeature[j].pos.x - tmpFeature[i].pos.x, 2) + pow(tmpFeature[j].pos.y - tmpFeature[i].pos.y, 2);
					if (tmpFeature[i].r > ED)
						tmpFeature[i].r = ED;
				}
			}

			auto r_val_compare = [&](const ResponseInfo&f1, const ResponseInfo& f2) {
				return f1.r > f2.r;
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

int Feature::maxFeatureSize = 128;

class FeatureMatch {
private:
	/**
	 *@param matchingIndex[in, out]
	 */
	static void CheckManyToOne(const FeatureInfo &f1, const FeatureInfo& f2, std::vector<Vec2i>& matchingIndex) {
		std::vector<int> f1_matches(f1.pos.size(), -1);
		for (int i = 0; i < matchingIndex.size(); i++) {
			if (f1_matches[matchingIndex[i].y] != -1) {
				cv::Mat tmp_f0(f1.descs[f1_matches[matchingIndex[i].y]]);
				cv::Mat tmp_f1(f1.descs[matchingIndex[i].x]);
				cv::Mat tmp_f2(f2.descs[matchingIndex[i].y]);

				float d1 = cv::norm(tmp_f0, tmp_f2, cv::NORM_L2);
				float d2 = cv::norm(tmp_f1, tmp_f2, cv::NORM_L2);

				if (d2 < d1) {
					f1_matches[matchingIndex[i].y] = matchingIndex[i].x;
				}
			}
			else {
				f1_matches[matchingIndex[i].y] = matchingIndex[i].x;
			}
		}

		std::vector<Vec2i> tmpMatchIndex(0);
		tmpMatchIndex.reserve(f1_matches.size());
		for (int i = 0; i < f1_matches.size(); i++) {
			if (f1_matches[i] != -1) {
				tmpMatchIndex.push_back(Vec2i(f1_matches[i], i));
			}
		}

		matchingIndex.assign(tmpMatchIndex.begin(), tmpMatchIndex.end());
	}
public:
	static std::vector<std::vector<Vec2i>> Match(const std::vector<FeatureInfo> &featureInfoList, float threshold = 0.8f) {
		std::vector<std::vector<Vec2i>> matches(0);
		matches.reserve(featureInfoList.size() - 1);
		for (int i = 0; i < featureInfoList.size() - 1; i++) {
			const FeatureInfo& f1 = featureInfoList[i];
			const FeatureInfo& f2 = featureInfoList[i + 1];

			const auto& f1_desc = f1.descs;
			const auto& f2_desc = f2.descs;

			std::vector<Vec2i> matchingIndex;

			for (int d1i = 0; d1i < f1_desc.size(); d1i++) {
				const cv::Mat feature1(f1_desc[d1i]);

				int lowIndex = 0;
				float lowDis = std::numeric_limits<float>::max();
				int secondIndex = 0;
				float secondDis = std::numeric_limits<float>::max();

				for (int d2i = 0; d2i < f2_desc.size(); d2i++) {
					const cv::Mat feature2(f2_desc[d2i]);

					float dist = cv::norm(feature1, feature2, cv::NORM_L2);
					if (dist < lowDis) {
						secondIndex = lowIndex;
						secondDis = lowDis;
						lowIndex = d2i;
						lowDis = dist;
					}
					else if (dist < secondDis) {
						secondIndex = d2i;
						secondDis = dist;
					}
				}
				// std::cout << lowDis / secondDis << std::endl;
				if (lowDis / secondDis < threshold) {
					matchingIndex.push_back(Vec2i(d1i, lowIndex));
				}
			}

			//std::cout << matchingIndex.size() << std::endl;
			//CheckManyToOne(f1, f2, matchingIndex);
			//std::cout << matchingIndex.size() << std::endl;
			matches.push_back(matchingIndex);
		}

		return matches;
	}
};

static std::random_device rd;

class ImageMatcher {
public:
	static std::vector<Vec2i> Match(const std::vector<cv::Mat> &images,const std::vector<FeatureInfo> &featureInfoList, const std::vector<std::vector<Vec2i>> &matches) {
		std::vector<Vec2i> alignments(0);
		alignments.reserve(matches.size());

		for (int i = 0; i < matches.size(); i++) {
			const auto& match = matches[i];

			const auto& feaPos1 = featureInfoList[i].pos;
			const auto& feaPos2 = featureInfoList[i + 1].pos;

			float minDifference = std::numeric_limits<float>::max();

			Vec2i alignment;
			//int K = (match.size() - 1) * match.size() / 2;
			for (int sample = 0; sample < match.size(); sample++) {
			//for (int j = 0; j < K; j++) {
				// 依照某一筆資料做位移，然後比對位移過後的feature point誤差
				//std::default_random_engine generator = std::default_random_engine(rd());
				//std::uniform_int_distribution<int> distribution(0, match.size() - 1);
				//int sample = distribution(generator);

				const Vec2i& sampleMatching = match[sample];
				Vec2i offset(0, images[i].cols);
				Vec2i offsetPoint2 = feaPos2[sampleMatching.y] + offset;

				Vec2i sampleAlignment = feaPos1[sampleMatching.x] - offsetPoint2;
				float sampleDist2 = sampleAlignment.x * sampleAlignment.x + sampleAlignment.y * sampleAlignment.y;

				// neglect bad matching which distance is larger
				// than image width
				if (sampleDist2 > images[i].cols * images[i].cols)
					continue;

				float difference = 0.0f;
				for (const auto& pair : match) {
					Vec2i moveP2 = feaPos2[pair.y] + offset + sampleAlignment;
					Vec2i pointDiff = feaPos1[pair.x] - moveP2;

					float dist2 = pointDiff.x * pointDiff.x + pointDiff.y * pointDiff.y;
					if (dist2 < images[i].cols * images[i].cols)
						difference += std::sqrt(dist2);
				}

				if (difference < minDifference) {
					minDifference = difference;
					alignment = sampleAlignment;
				}
			}

			alignments.push_back(alignment);
		}

		return alignments;
	}
};

class ImageBlender {
public:
	static cv::Mat Blend(const std::vector<cv::Mat>& images, const std::vector<Vec2i>& alignments, const bool ifAdjest) {
		std::vector<cv::Point> accumulateAlignments;
		accumulateAlignments.reserve(alignments.size());
		// swap x and y
		for (const Vec2i& align : alignments) 
			accumulateAlignments.push_back(cv::Point(align.y, align.x));

		int minDy = accumulateAlignments[0].y < 0 ? accumulateAlignments[0].y : 0;
		int maxDy = accumulateAlignments[0].y > 0 ? accumulateAlignments[0].y : 0;
		for (int i = 1; i < accumulateAlignments.size(); i++) {
			accumulateAlignments[i] += accumulateAlignments[i - 1];

			minDy = (accumulateAlignments[i].y < minDy) ? accumulateAlignments[i].y : minDy;
			maxDy = (accumulateAlignments[i].y > maxDy) ? accumulateAlignments[i].y : maxDy;
		}

		int allWidth = 0;
		int allHeight = 0;
		std::vector<int> offsetX;
		offsetX.reserve(images.size());

		for (auto& img : images) {
			allWidth += img.cols;
			offsetX.push_back(allWidth);
		}
		allWidth += accumulateAlignments[alignments.size() - 1].x;

		allHeight = images[0].rows;
		allHeight += (minDy < 0) ? -minDy : 0;
		allHeight += (maxDy > 0) ? maxDy : 0;

		cv::Mat panorama = cv::Mat::zeros(cv::Size(allWidth, allHeight), CV_32FC3);
		cv::Mat panoramaIndex = cv::Mat::zeros(panorama.size(), CV_8UC1);

		for (int n = 0; n < images.size(); n++) {
			const cv::Mat& image = images[n];

			const int beginX = (n == 0) ? 0 : offsetX[n - 1] + accumulateAlignments[n - 1].x;
			const int beginY = (n == 0) ? -minDy : -minDy + accumulateAlignments[n - 1].y;

			/*
				From the second image, we need to use origin alignment
				to build x-linear blending weight
			*/
			float intersectionRegion = (n > 0) ? -alignments[n - 1].y : 0.0f;

			for (int iy = 0; iy < image.rows; iy++) {
				for (int ix = 0; ix < image.cols; ix++) {
					int ry = iy + beginY, rx = ix + beginX;

					const cv::Vec3f originValue = panorama.at<cv::Vec3f>(ry, rx);
					const cv::Vec3f addValue = cv::Vec3f(image.at<cv::Vec3b>(iy, ix));

					if (panoramaIndex.at<uchar>(ry, rx) == 0) {
						panoramaIndex.at<uchar>(ry, rx) = 1;
						panorama.at<cv::Vec3f>(ry, rx) = addValue;
					}
					else {
						const float addWeight = ix / intersectionRegion;
						panorama.at<cv::Vec3f>(ry, rx) = (1.0f - addWeight) * originValue + addWeight * addValue;
					}
				}
			}
		}

		panorama.convertTo(panorama, CV_8UC3);

		if (ifAdjest) {
			int width = panorama.cols;
			int height = panorama.rows;

			int min_used_x = std::numeric_limits<int>::max();
			int max_used_x = -1;
			int min_used_y = std::numeric_limits<int>::max();
			int max_used_y = -1;

			// crop image
			panorama = panorama(cv::Rect(min_used_y, min_used_x, max_used_y, max_used_x));
		}

		return panorama;
	}
};
