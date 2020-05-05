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
// odd
#define WINDOW_SIZE 21
#define GaussWeight std::vector<std::vector<float>>

class Gaussian {
public:
	// w(x, y) = exp(-(x^2 + y^2) / (2 * variance))
	static GaussWeight genGaussWeight(int window_size) {
		GaussWeight weights(0);
		weights.reserve(window_size);
		int ori_x, ori_y;
		ori_x = ori_y = (window_size - 1) / 2;
		float sum = 0;
		float sigma = 1;
		float Q = 2 * sigma * sigma;
		for (int i = 0; i < window_size; i++) {
			std::vector<float> tpmw(0);
			tpmw.reserve(window_size);
			int offset_y = i - ori_y;
			for (int j = 0; j < window_size; j++) {
				int offset_x = j - ori_x;
				float gauss_coeff = std::exp(-(offset_y * offset_y + offset_x * offset_x) / Q) / (Q * 3.1415926);
				sum += gauss_coeff;
				tpmw.push_back(gauss_coeff);
			}
			weights.push_back(tpmw);
		}

		// normalize
		for (int i = 0; i < window_size; i++) {
			for (int j = 0; j < window_size; j++) {
				weights[i][j] /= sum;
				//printf("%f ", weights[i][j]);
			}
			//printf("\n");
		}
		return weights;
	}

	static std::vector<std::vector<float>> filter(const std::vector<std::vector<float>> &img, const GaussWeight &weights) {
		int half_x = 0, half_y = 0;
		half_x = half_y = (WINDOW_SIZE - 1) / 2;
		std::vector<std::vector<float>> f(img.size());
		for (int i = 0; i < img.size(); i++) {
			f[i].reserve(img[i].size());
			for (int j = 0; j < img[i].size(); j++) {
				float sum = 0;
				for (int x = -half_x, im_x = j - half_x; im_x <= j + half_x && im_x < img[i].size(); x++, im_x++) {
					if (im_x < 0)continue;
					for (int y = -half_y, im_y = i - half_y; im_y <= i + half_y && im_y < img.size(); y++, im_y++) {
						if (im_y < 0)continue;
						sum += img[im_y][im_x] * weights[half_y + y][half_x + x];
					}
				}
				f[i].push_back(sum);
			}
		}
		return f;
	}

	static std::vector<std::vector<Vec2f>> filter(const std::vector<std::vector<Vec2f>> &img, const GaussWeight &weights) {
		int half_x = 0, half_y = 0;
		half_x = half_y = (WINDOW_SIZE - 1) / 2;
		std::vector<std::vector<Vec2f>> f(img.size());
		for (int i = 0; i < img.size(); i++) {
			f[i].reserve(img[i].size());
			for (int j = 0; j < img[i].size(); j++) {
				Vec2f sum;
				for (int x = -half_x, im_x = j - half_x; im_x <= j + half_x && im_x < img[i].size(); x++, im_x++) {
					if (im_x < 0)continue;
					for (int y = -half_y, im_y = i - half_y; im_y <= i + half_y && im_y < img.size(); y++, im_y++) {
						if (im_y < 0)continue;
						sum += img[im_y][im_x] * weights[half_y + y][half_x + x];
					}
				}
				//std::cout << sum << std::endl;
				f[i].push_back(sum);
			}
		}
		return f;
	}

	static cv::Mat filter(const cv::Mat &img, const GaussWeight &weights) {
		cv::Mat blur(img);
		int half_x = 0, half_y = 0;
		half_x = half_y = (WINDOW_SIZE - 1) / 2;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				float sum = 0;
				for (int x = -half_x, im_x = j - half_x; im_x <= j + half_x && im_x < img.cols; x++, im_x++) {
					if (im_x < 0)continue;
					for (int y = -half_y, im_y = i - half_y; im_y <= i + half_y && im_y < img.rows; y++, im_y++) {
						if (im_y < 0)continue;
						sum += img.at<uchar>(im_y, im_x) * weights[half_y + y][half_x + x];
					}
				}
				blur.at<uchar>(i, j) = sum;
			}
		}
		return blur;
	}

};

class GradientImage {
public:
	std::vector<std::vector<Vec2f>> gradient;

	std::vector<std::vector<float>> grad_xx;
	std::vector<std::vector<float>> grad_yy;
	std::vector<std::vector<float>> grad_xy;

	std::vector<std::vector<float>> gauss_grad_xx;
	std::vector<std::vector<float>> gauss_grad_yy;
	std::vector<std::vector<float>> gauss_grad_xy;

	GradientImage(const cv::Mat &img, GaussWeight *weights = NULL) {
		bool ifDeletePointer = false;
		if (weights == NULL) {
			ifDeletePointer = true;
			weights = new GaussWeight(Gaussian::genGaussWeight(WINDOW_SIZE));
		}
		//for (auto l1 : *weights) {
		//	for (auto l2 : l1) {
		//		cout << l2 << " ";
		//	}
		//	cout << "\n";
		//}
		cv::Mat grayImg;
		// cvtColor : src會被釋放?
		cv::cvtColor(img, grayImg, CV_BGR2GRAY);
#ifdef WRITE_MIDDLE_STEP_IMAGE
		cv::imwrite("./Feature/gray_image.png", grayImg);
#endif // WRITE_MIDDLE_STEP_IMAGE

		grayImg = Gaussian::filter(grayImg, *weights);

#ifdef WRITE_MIDDLE_STEP_IMAGE
		cv::imwrite("./Feature/blur_gray_image.png", grayImg);
#endif // WRITE_MIDDLE_STEP_IMAGE

		gradient.reserve(grayImg.rows);
		// rows : height
		// cols : width
		for (int i = 0; i < grayImg.rows; i++) {
			std::vector<Vec2f> tmp;
			tmp.reserve(grayImg.cols);
			for (int j = 0; j < grayImg.cols; j++) {
				Vec2f g;
				//// x direction
				//if (j - 1 >= 0)g.x -= grayImg.at<uchar>(i, j - 1);
				//if (j + 1 < grayImg.cols)g.x += grayImg.at<uchar>(i, j + 1);
				//// y direction
				//if (i - 1 >= 0)g.y -= grayImg.at<uchar>(i - 1, j);
				//if (i + 1 < grayImg.rows)g.y += grayImg.at<uchar>(i + 1, j);

				// x direction
				g.x -= grayImg.at<uchar>(i, j);
				if (j + 1 < grayImg.cols)g.x += grayImg.at<uchar>(i, j + 1);
				else { 
					g.x += grayImg.at<uchar>(i, j - 1); 
					g.x *= -1;
				}
				// y direction
				g.y -= grayImg.at<uchar>(i, j);
				if (i + 1 < grayImg.rows)g.y += grayImg.at<uchar>(i + 1, j);
				else {
					g.y += grayImg.at<uchar>(i - 1, j);
					g.y *= -1;
				}
				tmp.push_back(g);
			}
			gradient.push_back(tmp);
		}

#ifdef WRITE_MIDDLE_STEP_IMAGE
		// for output gradient
		cv::Mat grad_x_img(grayImg.rows, grayImg.cols, CV_8U);
		cv::Mat grad_y_img(grayImg.rows, grayImg.cols, CV_8U);
		for (int i = 0; i < grayImg.rows; i++) {
			for (int j = 0; j < grayImg.cols; j++) {
				grad_x_img.at<uchar>(i, j) = std::abs(gradient[i][j].x);
				grad_y_img.at<uchar>(i, j) = std::abs(gradient[i][j].y);
			}
		}
		cv::imwrite("./Feature/grad_x_img.png", grad_x_img);
		cv::imwrite("./Feature/grad_y_img.png", grad_y_img);
#endif // WRITE_MIDDLE_STEP_IMAGE

		int indexNow = 0;
		grad_xx.resize(gradient.size());
		grad_yy.resize(gradient.size());
		grad_xy.resize(gradient.size());
		for (const std::vector<Vec2f> &row : gradient) {
			grad_xx[indexNow].reserve(row.size());
			grad_yy[indexNow].reserve(row.size());
			grad_xy[indexNow].reserve(row.size());
			for (const Vec2f &grad : row) {
				grad_xx[indexNow].push_back(grad.x*grad.x);
				grad_yy[indexNow].push_back(grad.y*grad.y);
				grad_xy[indexNow].push_back(grad.x*grad.y);
				//printf("%f ,%f ,%f\n", grad.x*grad.x, grad.y*grad.y, grad.x*grad.y);
			}
			indexNow++;
		}

		gauss_grad_xx = Gaussian::filter(grad_xx, *weights);
		gauss_grad_yy = Gaussian::filter(grad_yy, *weights);
		gauss_grad_xy = Gaussian::filter(grad_xy, *weights);

		if (ifDeletePointer)
			delete weights;
	}
};

#define FeatureList std::vector<std::vector<Vec2i>>

class Feature {
public:
	static FeatureList HarrisCorner(const std::vector<cv::Mat> &images, float threshold = 10000) {
		GaussWeight weightList = Gaussian::genGaussWeight(WINDOW_SIZE);
		FeatureList featurePoints(0);
		featurePoints.reserve(images.size());
		int half_x = 0, half_y = 0;
		half_x = half_y = (WINDOW_SIZE - 1) / 2;
		float k = 0.05;
		for (const cv::Mat &img : images) {
			std::vector<Vec2i> featureOfimg(0);
			featureOfimg.reserve(128);
			GradientImage gImg(img, &weightList);
			printf("rows : %d, cols : %d\n", img.rows, img.cols);
			for (int i = half_y; i < img.rows - half_y; i+= 1) {
				for (int j = half_x; j < img.cols - half_x; j += 1) {
					float denominator = gImg.gauss_grad_xx[i][j] + gImg.gauss_grad_yy[i][j];
					float R = gImg.gauss_grad_xx[i][j] * gImg.gauss_grad_yy[i][j] - gImg.gauss_grad_xy[i][j] * gImg.gauss_grad_xy[i][j] - k * denominator * denominator;
					if (R > threshold) {
						featureOfimg.push_back(Vec2i(i, j));
						//printf("%f\n", R);
					}
				}
			}
			featureOfimg.resize(featureOfimg.size());
			featurePoints.push_back(featureOfimg);
		}
		return featurePoints;
	}
};
