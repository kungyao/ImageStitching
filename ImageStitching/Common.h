#pragma once
#include <iostream>
#include <vector>

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
		cout << "(x : " << vec.x << ", y : " << vec.y << ")";
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