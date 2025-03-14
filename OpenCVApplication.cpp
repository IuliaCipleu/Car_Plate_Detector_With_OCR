// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <random>
#pragma comment(lib, "comdlg32.lib")  // for GetOpenFileNameA
#pragma comment(lib, "shell32.lib")   // for SHGetPathFromIDListA and SHBrowseForFolderA
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <regex>
#include <iostream>
#include <fstream>
#include <numeric>
#include <ctime>
#include <cstdlib>
#include <algorithm>  
#include <opencv2/ml.hpp>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

using namespace cv;
#include <filesystem>
namespace fs = std::filesystem;

const double MIN_ASPECT_RATIO = 2.5;
const double MAX_ASPECT_RATIO = 5.0;
bool show = true;

wchar_t* projectPath;

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
	waitKey();
}

void computeHistogram(const Mat& image, int* hist, const int hist_cols) {
	for (int i = 0; i < hist_cols; ++i) {
		hist[i] = 0;
	}

	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			int intensity = (int)image.at<uchar>(y, x);
			hist[intensity]++;
		}
	}
}

void computePDF(const int* hist, double* pdf, const int hist_cols, int n) {
	for (int i = 0; i < hist_cols; ++i) {
		pdf[i] = (double)hist[i] / n;
	}
}

void computeHistogramWithBins(const Mat& image, int* hist, const int hist_cols, const int m) {
	for (int i = 0; i < hist_cols; ++i) {
		hist[i] = 0;
	}
	int D = 256 / m;
	printf("%d\n", D);
	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			int intensity = (int)image.at<uchar>(y, x);
			hist[intensity / D]++;
		}
	}
}


void RGBToGrayscale(const Mat_<Vec3b>& src, Mat& dst) {
	int height = src.rows;
	int width = src.cols;

	dst = Mat(height, width, CV_8UC1);

	// Accessing individual pixels in a RGB 24 bits/pixel image
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b v3 = src.at<Vec3b>(i, j);
			uchar b = v3[0];
			uchar g = v3[1];
			uchar r = v3[2];
			dst.at<uchar>(i, j) = static_cast<uchar>((r + g + b) / 3);
		}
	}
}

Mat bilateralFilterAlgorithm(const Mat& src, int diam, double sigmaColor, double sigmaSpace) {
	Mat dst = Mat::zeros(src.size(), src.type());
	int radius = diam / 2;
	double colorCoef = -0.5 / (sigmaColor * sigmaColor);
	double spaceCoef = -0.5 / (sigmaSpace * sigmaSpace);

	int h = src.rows;
	int w = src.cols;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			double sumR = 0, sumG = 0, sumB = 0;
			double totalWeight = 0;
			double sumIntensity = 0;

			for (int dx = -radius; dx <= radius; dx++) {
				for (int dy = -radius; dy <= radius; dy++) {
					int x = i + dx;
					int y = j + dy;

					if (x >= 0 && x < h && y >= 0 && y < w) {
						uchar intensity = src.at<uchar>(x, y);
						uchar centerIntensity = src.at<uchar>(i, j);
						double r = sqrt(dx * dx + dy * dy);
						double colorDist = abs(intensity - centerIntensity); // Use absolute difference for grayscale images
						double weight = exp(-colorDist * colorDist * colorCoef - r * r * spaceCoef);
						sumIntensity += weight * intensity;
						totalWeight += weight;
					}
				}
			}
			// is 0-255?
			uchar avgIntensity = 0;
			if (totalWeight > 0) {
				avgIntensity = saturate_cast<uchar>(sumIntensity / totalWeight);
			}
			dst.at<uchar>(i, j) = avgIntensity;
		}
	}
	return dst;
}

Mat dilationHorizontal(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	Mat object(3, 3, CV_8UC1, Scalar(255));
	object.at < uchar>(1, 0) = 0;
	object.at < uchar>(1, 1) = 0;
	object.at < uchar>(1, 2) = 0;
	//for some works with vertical, some with horizontal
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						if (object.at<uchar>(k, l) == 0) {
							int x = i + k - (object.rows / 2);
							int y = j + l - (object.cols / 2);
							if (x >= 0 && x < h && y >= 0 && y < w) dst.at<uchar>(x, y) = 0;
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat dilationVertical(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	Mat object(3, 3, CV_8UC1, Scalar(255));
	object.at < uchar>(0, 1) = 0;
	object.at < uchar>(1, 1) = 0;
	object.at < uchar>(2, 1) = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						if (object.at<uchar>(k, l) == 0) {
							int x = i + k - (object.rows / 2);
							int y = j + l - (object.cols / 2);
							if (x >= 0 && x < h && y >= 0 && y < w) dst.at<uchar>(x, y) = 0;
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat dilationCross(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	Mat object(3, 3, CV_8UC1, Scalar(255));
	object.at < uchar>(0, 1) = 0;
	object.at < uchar>(1, 0) = 0;
	object.at < uchar>(1, 1) = 0;
	object.at < uchar>(1, 2) = 0;
	object.at < uchar>(2, 1) = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						if (object.at<uchar>(k, l) == 0) {
							int x = i + k - (object.rows / 2);
							int y = j + l - (object.cols / 2);
							if (x >= 0 && x < h && y >= 0 && y < w) dst.at<uchar>(x, y) = 0;
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat dilationGrayscale(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			uchar maxVal = 0;
			for (int k = 0; k < 8; k++) {
				int x = i + di[k];
				int y = j + dj[k];
				if (x >= 0 && x < h && y >= 0 && y < w) {
					maxVal = max(maxVal, src.at<uchar>(x, y));
				}
			}
			dst.at<uchar>(i, j) = maxVal;
		}
	}
	return dst;
}

Mat repeatDilationHorizontal(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = dilationHorizontal(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat repeatDilationVertical(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = dilationVertical(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat repeatDilationCross(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = dilationCross(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat erosion(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	Mat object(3, 3, CV_8UC1, Scalar(0));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						int x = i + k - (object.rows / 2);
						int y = j + l - (object.cols / 2);
						if (x >= 0 && x < h && y >= 0 && y < w) {
							if (object.at<uchar>(k, l) == 0 && src.at<uchar>(x, y) == 255) {
								dst.at<uchar>(i, j) = 255;
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat erosionCross(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	Mat object(3, 3, CV_8UC1, Scalar(255));
	object.at < uchar>(0, 1) = 0;
	object.at < uchar>(1, 0) = 0;
	object.at < uchar>(1, 1) = 0;
	object.at < uchar>(1, 2) = 0;
	object.at < uchar>(2, 1) = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						int x = i + k - (object.rows / 2);
						int y = j + l - (object.cols / 2);
						if (x >= 0 && x < h && y >= 0 && y < w) {
							if (object.at<uchar>(k, l) == 0 && src.at<uchar>(x, y) == 255) {
								if (x >= 0 && x < h && y >= 0 && y < w) dst.at<uchar>(i, j) = 255;
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat erosionVertical(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	Mat object(3, 3, CV_8UC1, Scalar(255));
	object.at < uchar>(0, 1) = 0;
	object.at < uchar>(1, 1) = 0;
	object.at < uchar>(2, 1) = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						int x = i + k - (object.rows / 2);
						int y = j + l - (object.cols / 2);
						if (x >= 0 && x < h && y >= 0 && y < w) {
							if (object.at<uchar>(k, l) == 0 && src.at<uchar>(x, y) == 255) {
								if (x >= 0 && x < h && y >= 0 && y < w) dst.at<uchar>(i, j) = 255;
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat erosionCrossGrayscale(const Mat& src) {
	Mat dst = src.clone();
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int h = src.rows;
	int w = src.cols;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			uchar minVal = 255;
			for (int k = 0; k < 8; k++) {
				int x = i + di[k];
				int y = j + dj[k];
				if (x >= 0 && x < h && y >= 0 && y < w) {
					minVal = min(minVal, src.at<uchar>(x, y));
				}
			}
			dst.at<uchar>(i, j) = minVal;
		}
	}
	return dst;
}


Mat repeatErosion(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = erosion(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat repeatErosionCross(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = erosionCross(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat opening(const Mat& src) {
	Mat erosionM = erosionCross(src);
	Mat dst = dilationCross(erosionM);
	return dst;
}

Mat repeatOpening(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = opening(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat closing(const Mat& src) {
	Mat dilationM = dilationCross(src);
	Mat dst = erosionCross(dilationM);
	return dst;
}

Mat closingVertical(const Mat& src) {
	Mat dilationM = dilationVertical(src);
	Mat dst = erosionVertical(dilationM);
	return dst;
}

Mat closingGrayscale(const Mat& src) {
	Mat dilationM = dilationGrayscale(src);
	Mat dst = erosionCrossGrayscale(dilationM);
	return dst;
}

Mat repeatClosing(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = closing(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat repeatClosingVertical(const Mat& src, int n) {
	Mat dst = src.clone();
	int i;
	Mat tempDst = dst.clone();
	for (i = 0; i < n; i++) {
		tempDst = closingVertical(dst);
		dst = tempDst.clone();
	}
	return dst;
}

Mat boundaryExtraction(const Mat& src) {
	Mat erosionM = erosion(src);
	Mat dst = src.clone();
	for (int i = 0; i < erosionM.rows; i++) {
		for (int j = 0; j < erosionM.cols; j++) {
			if (src.at<uchar>(i, j) == erosionM.at<uchar>(i, j)) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	return dst;
}

Mat regionFilling(const Mat& src) {
	Mat complement = src.clone();
	int h = src.rows;
	int w = src.cols;
	Point P0;
	P0.x = w / 2;
	P0.y = h / 2;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0) {
				complement.at<uchar>(i, j) = 255;
			}
			else {
				complement.at<uchar>(i, j) = 0;
			}
		}
	}
	Mat X = Mat(h, w, CV_8UC1, Scalar(255));
	X.at<uchar>(P0) = 0;
	Mat Xnext = X.clone();
	int n = 0;
	do {
		X = Xnext.clone();
		Mat dil = dilationCross(X);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (complement.at<uchar>(i, j) == 0 && dil.at<uchar>(i, j) == 0) {
					Xnext.at<uchar>(i, j) = 0;
				}
				else {
					Xnext.at<uchar>(i, j) = 255;
				}
			}
		}
	} while (cv::countNonZero(X != Xnext) > 0);

	Mat dst = Mat(h, w, CV_8UC1, Scalar(255));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (Xnext.at<uchar>(i, j) == 0 || src.at<uchar>(i, j) == 0) {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

void twoPassComponentLabeling(const Mat& src, std::vector<std::vector<Point>>& contours) {
	Mat labels(src.rows, src.cols, CV_32SC1, Scalar(0));
	int h = src.rows;
	int w = src.cols;
	int label = 0;
	int newLabel = 0;
	std::vector<std::vector<int>> edges(10000);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				std::vector<int> listNeighbors;
				if (i >= 1 && j >= 1 && labels.at<int>(i - 1, j - 1) > 0) {
					listNeighbors.push_back(labels.at<int>(i - 1, j - 1));
				}
				if (i >= 1 && labels.at<int>(i - 1, j) > 0) {
					listNeighbors.push_back(labels.at<int>(i - 1, j));
				}
				if (i >= 1 && j + 1 < w && labels.at<int>(i - 1, j + 1) > 0) {
					listNeighbors.push_back(labels.at<int>(i - 1, j + 1));
				}
				if (j >= 1 && labels.at<int>(i, j - 1) > 0) {
					listNeighbors.push_back(labels.at<int>(i, j - 1));
				}
				if (j + 1 < w && labels.at<int>(i, j + 1) > 0) {
					listNeighbors.push_back(labels.at<int>(i, j + 1));
				}
				if (i + 1 < h && j >= 1 && labels.at<int>(i + 1, j - 1) > 0) {
					listNeighbors.push_back(labels.at<int>(i + 1, j - 1));
				}
				if (i + 1 < h && labels.at<int>(i + 1, j) > 0) {
					listNeighbors.push_back(labels.at<int>(i + 1, j));
				}
				if (i + 1 < h && j + 1 < w && labels.at<int>(i + 1, j + 1) > 0) {
					listNeighbors.push_back(labels.at<int>(i + 1, j + 1));
				}
				if (listNeighbors.size() == 0) {
					label++;
					labels.at<int>(i, j) = label;
				}
				else {
					int minList = *min_element(listNeighbors.begin(), listNeighbors.end());
					labels.at<int>(i, j) = minList;
					for (int y : listNeighbors)
					{
						if (y != minList)
						{
							edges[minList].push_back(y);
							edges[y].push_back(minList);
						}
					}
				}
			}
		}
	}

	std::vector<int> newLabels(label + 1);
	for (int i = 0; i < h; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue <int> q;
			newLabels[i] = newLabel;
			q.push(i);
			while (!q.empty()) {
				int front = q.front();
				q.pop();
				for (int j : edges[front]) {
					if (newLabels[j] == 0) {
						newLabels[j] = newLabel;
						q.push(j);
					}
				}
			}
		}
	}

	contours.clear();
	contours.resize(newLabel);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int label = labels.at<int>(i, j);
			if (label != 0 && newLabels[label] > 0) {
				contours[newLabels[label] - 1].push_back(Point(j, i));
			}
		}
	}
}

Mat generalFilterSpatialDomain(const Mat& src, Mat& kernel) {
	Mat dst = src.clone();
	float scale = 1.0;
	float sumNeg = 0.0;
	float sumPos = 0.0;
	bool isLow = true;
	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			printf("%f ", kernel.at<float>(i, j));
			if (kernel.at<float>(i, j) >= 0) {
				sumPos = sumPos + kernel.at<float>(i, j);

			}
			else {
				sumNeg = sumNeg + (-1.0) * kernel.at<float>(i, j);
				isLow = false;
			}
		}
	}
	if (isLow) {
		scale = 1.0 / (sumPos + sumNeg);
		printf("Is low\n");
	}
	else {
		scale = 1 / (2 * max(sumPos, sumNeg));
		printf("Is high\n");
	}
	printf("Scale: %f\n", scale);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float sum = 0.0;
			for (int k = 0; k < kernel.rows; k++) {
				for (int l = 0; l < kernel.cols; l++) {
					int iPixel = i + k - kernel.rows / 2;
					int jPixel = j + l - kernel.cols / 2;
					if (iPixel >= 0 && iPixel < src.rows && jPixel >= 0 && jPixel < src.cols) {
						sum = sum + kernel.at<float>(k, l) * src.at<uchar>(iPixel, jPixel);
					}
				}
			}
			if (sumNeg == 0) {
				dst.at<uchar>(i, j) = scale * sum;
			}
			else {
				dst.at<uchar>(i, j) = scale * sum + (255 / 2);
			}
		}
	}

	return dst;
}

Mat gaussianFilterDecomposition(Mat& src, int w, Mat& intermediate) {
	Mat Gx = Mat(w, 1, CV_32FC1);
	Mat Gy = Mat(1, w, CV_32FC1);
	float sigma = (float)w / 6;
	for (int i = 0; i < w; i++) {
		Gx.at<float>(i, 0) = (float)(1 / sqrt(2 * PI) * sigma) * exp((-1) * (i - w / 2) * (i - w / 2) / (2 * sigma * sigma));
		Gy.at<float>(0, i) = (float)(1 / sqrt(2 * PI) * sigma) * exp((-1) * (i - w / 2) * (i - w / 2) / (2 * sigma * sigma));
	}
	intermediate = generalFilterSpatialDomain(src, Gx);
	Mat dst = generalFilterSpatialDomain(intermediate, Gy);
	return dst;
}

Mat createHorizontalKernel(int id) {
	int w;
	if (id == 0 || id == 1) {
		w = 3;
	}
	else {
		w = 2;
	}
	Mat kernel = Mat(w, w, CV_32FC1);
	int size = w * w;
	switch (id) {
	case 0: //Prewitt
		for (int i = 0; i < 3; i++) {
			kernel.at<float>(i, 0) = -1.0;
			kernel.at<float>(i, 1) = 0.0;
			kernel.at<float>(i, 2) = 1.0;
		}
		break;
	case 1: //Sobel
		for (int i = 0; i < 3; i++) {
			kernel.at<float>(i, 0) = -1.0;
			kernel.at<float>(i, 1) = 0.0;
			kernel.at<float>(i, 2) = 1.0;
		}
		kernel.at<float>(1, 0) = -2.0;
		kernel.at<float>(1, 2) = 2.0;
		break;
	case 2: //Robert (cross)
		kernel.at<float>(0, 0) = 1.0;
		kernel.at<float>(0, 1) = 0.0;
		kernel.at<float>(1, 0) = 0.0;
		kernel.at<float>(1, 1) = -1.0;
		break;
	default:
		break;
	}
	return kernel;
}

Mat createVerticalKernel(int id) {
	int w;
	if (id == 0 || id == 1) {
		w = 3;
	}
	else {
		w = 2;
	}
	Mat kernel = Mat(w, w, CV_32FC1);
	int size = w * w;
	switch (id) {
	case 0: //Prewitt
		for (int i = 0; i < 3; i++) {
			kernel.at<float>(0, i) = 1.0;
			kernel.at<float>(1, i) = 0.0;
			kernel.at<float>(2, i) = -1.0;
		}
		break;
	case 1: //Sobel
		for (int i = 0; i < 3; i++) {
			kernel.at<float>(0, i) = 1.0;
			kernel.at<float>(1, i) = 0.0;
			kernel.at<float>(2, i) = -1.0;
		}
		kernel.at<float>(0, 1) = 2.0;
		kernel.at<float>(2, 1) = -2.0;
		break;
	case 2: //Robert (cross)
		kernel.at<float>(0, 0) = 0.0;
		kernel.at<float>(0, 1) = -1.0;
		kernel.at<float>(1, 0) = 1.0;
		kernel.at<float>(1, 1) = 0.0;
		break;
	default:
		break;
	}
	return kernel;
}

Mat convolution(const Mat& src, Mat& kernel) {
	Mat dst = Mat::zeros(src.size(), CV_32FC1);
	float sumNeg = 0.0;
	float sumPos = 0.0;
	bool isLow = true;
	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			if (kernel.at<float>(i, j) >= 0) {
				sumPos += kernel.at<float>(i, j);
			}
			else {
				sumNeg += (-1.0) * kernel.at<float>(i, j);
				isLow = false;
			}
		}
	}

	float scale;
	if (isLow) {
		scale = 1.0 / (sumPos + sumNeg);
	}
	else {
		scale = 1 / (2 * max(sumPos, sumNeg));
	}

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float sum = 0.0;
			for (int k = 0; k < kernel.rows; k++) {
				for (int l = 0; l < kernel.cols; l++) {
					int iPixel = i + k - kernel.rows / 2;
					int jPixel = j + l - kernel.cols / 2;
					if (iPixel >= 0 && iPixel < src.rows && jPixel >= 0 && jPixel < src.cols) {
						sum += kernel.at<float>(k, l) * src.at<uchar>(iPixel, jPixel);
					}
				}
			}
			dst.at<float>(i, j) = sum;
		}
	}

	return dst;
}

Mat magnitude_mat(Mat& src, Mat& kernelX, Mat& kernelY) {
	Mat dst(src.rows, src.cols, CV_32FC1, Scalar(0));
	Mat gradientX = convolution(src, kernelX);
	Mat gradientY = convolution(src, kernelY);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float gx = gradientX.at<float>(i, j);
			float gy = gradientY.at<float>(i, j);
			dst.at<float>(i, j) = sqrt(gx * gx + gy * gy) / (4 * sqrt(2));
		}
	}
	return dst;
}

Mat direction_mat(Mat& src, Mat& kernelX, Mat& kernelY, bool isRobert) {
	Mat dst = Mat(src.rows, src.cols, CV_32FC1, Scalar(0, 0));
	Mat gradientX = convolution(src, kernelX);
	Mat gradientY = convolution(src, kernelY);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<float>(i, j) = atan2(gradientY.at<float>(i, j), gradientX.at<float>(i, j)) + PI;
			if (isRobert) {
				dst.at<float>(i, j) += 3 * PI / 4;
			}
		}
	}
	return dst;
}

Mat thresholding(Mat& src, int th) {
	Mat dst = Mat(src.rows, src.cols, CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<float>(i, j) < th) {
				dst.at<float>(i, j) = 0.0;
			}
			else dst.at<float>(i, j) = 255.0;
		}
	}
	return dst;
}

Mat thresholding2ParamUchar(Mat& src, int thL, int thH) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) < thL) {
				dst.at<uchar>(i, j) = 0;
			}
			else if (src.at<uchar>(i, j) > thH) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 128;
			}
		}
	}
	return dst;
}

void BFS_canny(Mat& src, Mat& labels, int i, int j, int label) {
	int h = src.rows;
	int w = src.cols;
	std::queue<Point> Q;
	Q.push(Point(j, i));
	while (!Q.empty()) {
		Point q = Q.front();
		Q.pop();
		if (src.at<uchar>(q.y, q.x) == 255) {
			labels.at<int>(q.y, q.x) = label;
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int nx = q.x + dx;
					int ny = q.y + dy;
					if (nx >= 0 && nx < w && ny >= 0 && ny < h && src.at<uchar>(ny, nx) == 128 && labels.at<int>(ny, nx) == 0) {
						labels.at<int>(ny, nx) = label;
						Q.push(Point(nx, ny));
					}
				}
			}
		}
	}
}

void BFS_canny2(Mat& src, Mat& labels, int i, int j, int label) {
	int h = src.rows;
	int w = src.cols;
	std::queue<Point> Q;
	Q.push(Point(j, i));
	while (!Q.empty()) {
		Point q = Q.front();
		Q.pop();
		labels.at<int>(q.y, q.x) = label;
		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				int nx = q.x + dx;
				int ny = q.y + dy;
				if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
					if (src.at<uchar>(ny, nx) == 255 && labels.at<int>(ny, nx) == 0) {
						labels.at<int>(ny, nx) = label;
						Q.push(Point(nx, ny));
					}
				}
			}
		}
	}
}

Mat cannyEdgeDetection(Mat& src, int w) {
	Mat intermediate = src.clone();
	Mat gaussianFiltered = gaussianFilterDecomposition(src, w, intermediate);
	if (show) {
		imshow("Gaussian Filter", gaussianFiltered);
	}
	Mat xSobel = createHorizontalKernel(1);
	Mat ySobel = createVerticalKernel(1);
	Mat magn = magnitude_mat(gaussianFiltered, xSobel, ySobel);
	Mat dir = direction_mat(gaussianFiltered, xSobel, ySobel, false);
	int h = src.rows;
	int w1 = src.cols;
	Mat magnUchar = Mat(h, w1, CV_8UC1);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w1; j++) {
			magnUchar.at<uchar>(i, j) = magn.at<float>(i, j);
		}
	}
	int neighborsX[] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	int neighborsY[] = { -1, -1, 1, 1, 1, 1, 0, -1 };
	Mat nonMaxima = Mat(h, w1, CV_8UC1, Scalar(0));
	int NoPixels = 0;
	int ZeroGradientModulePixels = 0;
	float p = 0.75;
	Mat magnTh = magn.clone();
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w1; j++) {
			int currentDir;
			int neighborI = i;
			int neighborJ = j;
			int neighborI2 = i;
			int neighborJ2 = j;
			if ((dir.at<float>(i, j) >= 3 * PI / 8 && dir.at<float>(i, j) < 5 * PI / 8) || (dir.at<float>(i, j) >= 11 * PI / 8 && dir.at<float>(i, j) < 13 * PI / 8)) {
				currentDir = 0;
				neighborI--;
				neighborI2++;
			}
			else {
				if ((dir.at<float>(i, j) >= PI / 8 && dir.at<float>(i, j) < 3 * PI / 8) || (dir.at<float>(i, j) >= 9 * PI / 8 && dir.at<float>(i, j) < 11 * PI / 8)) {
					currentDir = 1;
					neighborI--;
					neighborJ++;
					neighborI2++;
					neighborJ2--;
				}
				else {
					if ((dir.at<float>(i, j) >= 0 && dir.at<float>(i, j) < PI / 8) || (dir.at<float>(i, j) >= 7 * PI / 8 && dir.at<float>(i, j) < 9 * PI / 8) || (dir.at<float>(i, j) >= 15 * PI / 8 && dir.at<float>(i, j) < 2 * PI)) {
						currentDir = 2;
						neighborJ--;
						neighborJ2++;
					}
					else {
						currentDir = 3;
						neighborI--;
						neighborJ--;
						neighborI2++;
						neighborJ2++;
					}
				}
			}

			if (neighborI >= 0 && neighborI < h && neighborJ >= 0 && neighborJ < w1 &&
				neighborI2 >= 0 && neighborI2 < h && neighborJ2 >= 0 && neighborJ2 < w1) {
				if (magnUchar.at<uchar>(i, j) > magnUchar.at<uchar>(neighborI, neighborJ) && magnUchar.at<uchar>(i, j) > magnUchar.at<uchar>(neighborI2, neighborJ2)) {
					nonMaxima.at<uchar>(i, j) = magnUchar.at<uchar>(i, j);
					NoPixels++;
				}
				else nonMaxima.at<uchar>(i, j) = 0;
			}

			if (nonMaxima.at<uchar>(i, j) == 0 && i != 0 && j != 0 && i < h - 1 && j < w1 - 1) {
				ZeroGradientModulePixels++;
			}
		}
	}
	int hist[256] = { 0 };
	computeHistogram(nonMaxima, hist, 256);
	int NoNonEdge = p * ((h - 2) * (w1 - 2) - hist[0]);
	int NoEdgePixels = (1 - p) * ((h - 2) * (w1 - 2) - hist[0]);
	int sum = 0;
	int thresholdHigh = 255;
	for (int i = 255; i >= 0; i--) {
		sum = sum + hist[i];
		if (sum > NoNonEdge) {
			thresholdHigh = i;
			break;
		}
	}
	float k = 0.55;
	int thresholdLow = k * thresholdHigh;
	Mat adaptiveThresholding = thresholding2ParamUchar(nonMaxima, thresholdLow, thresholdHigh);
	Mat dst = adaptiveThresholding.clone();

	Mat labels(h, w1, CV_32SC1, Scalar(0));
	int label = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w1; j++) {
			if (dst.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0) {
				label++;
				BFS_canny2(dst, labels, i, j, label);
			}
		}
	}

	Mat binaryEdges = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (labels.at<int>(i, j) > 0) {
				binaryEdges.at<uchar>(i, j) = 255;
			}
		}
	}
	return binaryEdges;
}

bool isBoundaryPixel(const Mat& image, int x, int y) {
	if (x <= 0 || x >= image.cols - 1 || y <= 0 || y >= image.rows - 1)
		return true;

	return (image.at<uchar>(y, x) == 0 &&
		(image.at<uchar>(y, x - 1) == 255 || image.at<uchar>(y, x + 1) == 255 ||
			image.at<uchar>(y - 1, x) == 255 || image.at<uchar>(y + 1, x) == 255));
}

void findBoundary(Mat& image, std::vector<Point>& contour, int x, int y, int maxSize) {
	if (x < 0 || x >= image.cols || y < 0 || y >= image.rows || image.at<uchar>(y, x) != 0 || contour.size() >= maxSize)
		return;

	contour.push_back(Point(x, y));
	image.at<uchar>(y, x) = 255;

	if (isBoundaryPixel(image, x + 1, y)) findBoundary(image, contour, x + 1, y, maxSize);
	if (isBoundaryPixel(image, x - 1, y)) findBoundary(image, contour, x - 1, y, maxSize);
	if (isBoundaryPixel(image, x, y + 1)) findBoundary(image, contour, x, y + 1, maxSize);
	if (isBoundaryPixel(image, x, y - 1)) findBoundary(image, contour, x, y - 1, maxSize);
}

std::vector<std::vector<Point>> myFindContours(const Mat& image) {
	std::vector<std::vector<Point>> contours;

	Mat imageCopy = image.clone();

	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			if (imageCopy.at<uchar>(i, j) == 0 && isBoundaryPixel(imageCopy, j, i)) {
				std::vector<Point> contour;
				int maxSize = 10000;
				findBoundary(imageCopy, contour, j, i, maxSize);
				contours.push_back(contour);
			}
		}
	}

	return contours;
}

Rect boundingRect(const std::vector<Point>& contour) {
	int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
	for (const Point& p : contour) {
		minX = min(minX, p.x);
		minY = min(minY, p.y);
		maxX = max(maxX, p.x);
		maxY = max(maxY, p.y);
	}
	return Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
}

bool checkBorderColor(const Mat& image, const Rect& rect, double colorDifferenceThreshold) {
	Rect intersection = rect & Rect(0, 0, image.cols, image.rows);
	if (intersection.area() <= 0) {
		return false;
	}

	Mat borderRegion = image(intersection);

	Scalar meanColor = mean(borderRegion);

	Mat diff;
	absdiff(borderRegion, meanColor, diff);
	Scalar diffMean = mean(diff);

	double avgDiff = (diffMean[0] + diffMean[1] + diffMean[2]) / 3.0;

	printf("Avg: %f\n", avgDiff);
	return avgDiff < colorDifferenceThreshold;
}

inline uchar reduceVal(const uchar val)
{
	if (val < 64) return 0;
	if (val < 128) return 64;
	return 255;
}

void processColors(Mat& img)
{
	uchar* pixelPtr = img.data;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			const int pi = i * img.cols * 3 + j * 3;
			pixelPtr[pi + 0] = reduceVal(pixelPtr[pi + 0]); // B
			pixelPtr[pi + 1] = reduceVal(pixelPtr[pi + 1]); // G
			pixelPtr[pi + 2] = reduceVal(pixelPtr[pi + 2]); // R
		}
	}
	if (show) {
		imshow("Reduced colors", img);
	}
	waitKey();
}

bool checkNumberOfColors(const Mat& image, const Rect& rect, int minColors, int maxColors) {
	Rect intersection = rect & Rect(0, 0, image.cols, image.rows);
	if (intersection.area() <= 0) {
		return false;
	}
	Mat borderRegion = image(intersection);

	Mat hsvBorderRegion;
	cvtColor(borderRegion, hsvBorderRegion, COLOR_BGR2HSV);
	Mat copy = borderRegion.clone();
	processColors(copy);

	struct Vec3bComparator {
		bool operator()(const Vec3b& lhs, const Vec3b& rhs) const {
			return (lhs[0] + lhs[1] + lhs[2]) < (rhs[0] + rhs[1] + rhs[2]);
		}
	};

	std::set<Vec3b, Vec3bComparator> uniqueColors;
	for (int i = 0; i < copy.rows; ++i) {
		for (int j = 0; j < copy.cols; ++j) {
			Vec3b pixelColor = copy.at<Vec3b>(i, j);
			uniqueColors.insert(pixelColor);
		}
	}

	printf("Number of unique colors: %zu\n", uniqueColors.size());
	return (uniqueColors.size() >= minColors && uniqueColors.size() <= maxColors);
}

Mat RGBToHSV(Mat& src) {

	int height = src.rows;
	int width = src.cols;

	Mat dstH = Mat(height, width, CV_8UC1);
	Mat dstS = Mat(height, width, CV_8UC1);
	Mat dstV = Mat(height, width, CV_8UC1);

	// Accessing individual pixels in a RGB 24 bits/pixel image
	// Inefficient way -> slow
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float r = (float)src.at<Vec3b>(i, j)[2] / 255;
			float g = (float)src.at<Vec3b>(i, j)[1] / 255;
			float b = (float)src.at<Vec3b>(i, j)[0] / 255;

			float mx = -1.0f;
			float mn = 1000.0f;
			mx = max(r, max(g, b));
			mn = min(r, min(g, b));

			float c = mx - mn;
			float v = mx;
			float s, h;
			if (v != 0) {
				s = c / v;
			}
			else s = 0;
			if (c != 0) {
				if (mx == r) {
					h = 60 * (g - b) / c;
				}
				if (mx == g) {
					h = 120 + 60 * (b - r) / c;
				}
				if (mx == b) {
					h = 240 + 60 * (r - g) / c;
				}
			}
			else
				h = 0;
			if (h < 0) h = h + 360;
			dstH.at<uchar>(i, j) = h * 255 / 360;
			dstS.at<uchar>(i, j) = s * 255;
			dstV.at<uchar>(i, j) = v * 255;
		}
	}
	Mat hsvImage;
	std::vector<Mat> channels = { dstH, dstS, dstV };
	merge(channels, hsvImage);

	return hsvImage;
}

bool isMostlyRed(const Mat& colorImage, const Rect& rect, double redThreshold) {
	Mat region = colorImage(rect);
	if (show) {
		imshow("Region in Color", region);
	}
	Mat hsv = RGBToHSV(region);
	int lower_red = 0; // lower bound of red hue
	int upper_red = 30; // upper bound of red hue

	Mat mask;
	inRange(hsv, Scalar(lower_red, 100, 100), Scalar(upper_red, 255, 255), mask);

	int red_pixel_count = countNonZero(mask);

	int total_pixels = region.rows * region.cols;

	double red_percentage = (double)red_pixel_count / total_pixels;
	printf("Percentage red: %f\n", red_percentage);
	return red_percentage > redThreshold;
}

Rect detectLicensePlate(const Mat& src, const Mat& colorImage) {
	Mat contoursImage = src.clone();
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	findContours(contoursImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (contours.empty()) {
		printf("No contours\n");
	}
	else {
		printf("Contours size : %d\n", contours.size());
	}

	Rect selectedPlateRect;
	double maxArea = 0;
	int imageCenterY = src.rows / 2;

	for (const auto& contour : contours) {
		double area = contourArea(contour);
		if (area > 800.0) {
			printf("Area: % f\n", area);
			Rect plateRect = boundingRect(contour);
			double aspectRatio = static_cast<double>(plateRect.width) / plateRect.height;
			if (aspectRatio > MIN_ASPECT_RATIO && aspectRatio < MAX_ASPECT_RATIO && checkBorderColor(colorImage, plateRect, 50.0) &&
				plateRect.y + plateRect.height > imageCenterY && checkNumberOfColors(colorImage, plateRect, 3, 10)) {
				printf("In if\n");
				maxArea = area;
				selectedPlateRect = plateRect;
				printf("Top-left: (%d, %d)\n", plateRect.x, plateRect.y);
				printf("Top-right: (%d, %d)\n", plateRect.x + plateRect.width, plateRect.y);
				printf("Bottom-left: (%d, %d)\n", plateRect.x, plateRect.y + plateRect.height);
				printf("Bottom-right: (%d, %d)\n", plateRect.x + plateRect.width, plateRect.y + plateRect.height);
				Mat detectedPlate = colorImage.clone();
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				if (show) {
					imshow("Detected Plate", detectedPlate);
				}
				waitKey();
			}
		}
	}
	return selectedPlateRect;
}

Rect detectLicensePlateWithRedComputation(const Mat& src, const Mat& colorImage) {
	Mat contoursImage = src.clone();
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	findContours(contoursImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Rect selectedPlateRect;
	double maxArea = 0;
	int imageCenterY = src.rows / 2;

	for (const auto& contour : contours) {
		double area = contourArea(contour);
		if (area > 800.0) {
			printf("Area: % f\n", area);
			Rect plateRect = boundingRect(contour);
			double aspectRatio = static_cast<double>(plateRect.width) / plateRect.height;
			if (aspectRatio > MIN_ASPECT_RATIO && aspectRatio < MAX_ASPECT_RATIO && checkBorderColor(colorImage, plateRect, 50.0) &&
				plateRect.y + plateRect.height > imageCenterY && checkNumberOfColors(colorImage, plateRect, 3, 10) && !isMostlyRed(colorImage, plateRect, 0.04)) {
				printf("In if\n");
				maxArea = area;
				selectedPlateRect = plateRect;
				printf("Top-left: (%d, %d)\n", plateRect.x, plateRect.y);
				printf("Top-right: (%d, %d)\n", plateRect.x + plateRect.width, plateRect.y);
				printf("Bottom-left: (%d, %d)\n", plateRect.x, plateRect.y + plateRect.height);
				printf("Bottom-right: (%d, %d)\n", plateRect.x + plateRect.width, plateRect.y + plateRect.height);
				Mat detectedPlate = colorImage.clone();
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				if (show) {
					imshow("Detected Plate", detectedPlate);
				}
				waitKey();
			}
		}
	}
	return selectedPlateRect;
}

Mat negativeTransform(Mat& src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			uchar val = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = 255 - val;
		}

	return dst;
}

Mat basicGlobalThresholding(const Mat& src) {
	Mat dst = src.clone();
	int hist[256];
	computeHistogram(src, hist, 255);
	int mn = 10000000;
	int mx = -1;
	for (int i = 0; i < 255; i++) {
		if (hist[i] < mn) {
			mn = i;
		}
		if (hist[i] > mx) {
			mx = i;
		}
	}
	double T = (double)(mn + mx) / 2;
	double Tnew = T;
	do {
		T = Tnew;
		int N1 = 0;
		for (int i = mn; i <= T; i++) {
			N1 = N1 + hist[i];
		}
		int sum1 = 0;
		for (int i = mn; i <= T; i++) {
			sum1 = sum1 + i * hist[i];
		}
		int g1 = 0;
		if (N1 != 0)  g1 = sum1 / N1;
		int N2 = 0;
		for (int i = T; i <= mx; i++) {
			N2 = N2 + hist[i];
		}
		int sum2 = 0;
		for (int i = T; i <= mx; i++) {
			sum2 = sum2 + i * hist[i];
		}
		int g2 = 0;
		if (N2 != 0) g2 = sum2 / N2;
		Tnew = (double)(g1 + g2) / 2;
	} while (abs(T - Tnew) > 0.1);
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst.at<uchar>(i, j) < Tnew) {
				dst.at<uchar>(i, j) = 0;
			}
			else dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

Mat adaptiveThresholding(Mat& src, int blockSize, double c) {
	Mat dst(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			int startX = max(0, j - blockSize / 2);
			int startY = max(0, i - blockSize / 2);
			int endX = min(src.cols - 1, j + blockSize / 2);
			int endY = min(src.rows - 1, i + blockSize / 2);

			int sum = 0;
			int count = 0;
			for (int y = startY; y <= endY; ++y) {
				for (int x = startX; x <= endX; ++x) {
					sum += src.at<uchar>(y, x);
					++count;
				}
			}

			double mean = static_cast<double>(sum) / count;
			if (src.at<uchar>(i, j) < mean - c)
				dst.at<uchar>(i, j) = 0;
			else
				dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

Mat findROI(const Mat& img) {
	Mat hist_vert, hist_hor;
	Mat row_sums(img.rows, 1, CV_32SC1);
	for (int y = 0; y < img.rows; y++) {
		int sum = 0;
		for (int x = 0; x < img.cols; x++) {
			sum += img.at<uchar>(y, x);
		}
		row_sums.at<int>(y, 0) = sum;
	}

	Mat col_sums(1, img.cols, CV_32SC1);
	for (int x = 0; x < img.cols; x++) {
		int sum = 0;
		for (int y = 0; y < img.rows; y++) {
			sum += img.at<uchar>(y, x);
		}
		col_sums.at<int>(0, x) = sum;
	}

	int left = 0, right = img.cols - 1, top = 0, bottom = img.rows - 1;
	for (int i = 0; i < hist_vert.cols; i++) {
		if (hist_vert.at<int>(0, i) > 0) {
			left = i;
			break;
		}
	}
	for (int i = hist_vert.cols - 1; i >= 0; i--) {
		if (hist_vert.at<int>(0, i) > 0) {
			right = i;
			break;
		}
	}
	for (int i = 0; i < hist_hor.rows; i++) {
		if (hist_hor.at<int>(i, 0) > 0) {
			top = i;
			break;
		}
	}
	for (int i = hist_hor.rows - 1; i >= 0; i--) {
		if (hist_hor.at<int>(i, 0) > 0) {
			bottom = i;
			break;
		}
	}

	Mat cropped_img = img(Rect(left, top, right - left + 1, bottom - top + 1));

	if (show) {
		imshow("Cropped Image", cropped_img);
	}
	return cropped_img;
}

Mat cutBorders(const Mat& binary, double percentageV, double percentageH) {
	Mat vertical_hist(1, binary.cols, CV_32S, Scalar(0));
	for (int col = 0; col < binary.cols; ++col) {
		vertical_hist.at<int>(0, col) = countNonZero(binary.col(col));
	}

	double maxVal;
	minMaxLoc(vertical_hist, nullptr, &maxVal);
	int thresholdVertically = static_cast<int>(maxVal * percentageV);

	int left = -1, right = -1;
	for (int col = 0; col < vertical_hist.cols; ++col) {
		if (vertical_hist.at<int>(0, col) > thresholdVertically) {
			if (left == -1) {
				left = col;
			}
			right = col;
		}
	}

	Mat horizontal_hist(binary.rows, 1, CV_32S, Scalar(0));
	for (int row = 0; row < binary.rows; ++row) {
		horizontal_hist.at<int>(row, 0) = countNonZero(binary.row(row));
	}

	int thresholdHorizontally = static_cast<int>(maxVal * percentageH);
	int top = -1, bottom = -1;
	for (int row = 0; row < horizontal_hist.rows; ++row) {
		if (horizontal_hist.at<int>(row, 0) > thresholdHorizontally) {
			if (top == -1) {
				top = row;
			}
			bottom = row;
		}
	}
	Mat license_plate = binary.clone();
	if (left != -1 && right != -1 && top != -1 && bottom != -1) {
		Rect roi(left, top, right - left, bottom - top);
		license_plate = binary(roi);
	}
	else {
		printf("License plate boundaries not found!\n");
	}
	return license_plate;
}

Mat cutBordersDynamic(const Mat& binary, double verticalThresholdRatio, double horizontalThresholdRatio) {
	// Compute vertical projection
	Mat vertical_hist(1, binary.cols, CV_32S, Scalar(0));
	for (int col = 0; col < binary.cols; ++col) {
		vertical_hist.at<int>(0, col) = countNonZero(binary.col(col));
	}

	// Compute horizontal projection
	Mat horizontal_hist(binary.rows, 1, CV_32S, Scalar(0));
	for (int row = 0; row < binary.rows; ++row) {
		horizontal_hist.at<int>(row, 0) = countNonZero(binary.row(row));
	}

	// Compute dynamic thresholds
	double maxVertical = *std::max_element(vertical_hist.begin<int>(), vertical_hist.end<int>());
	double maxHorizontal = *std::max_element(horizontal_hist.begin<int>(), horizontal_hist.end<int>());

	int verticalThreshold = static_cast<int>(maxVertical * verticalThresholdRatio);
	int horizontalThreshold = static_cast<int>(maxHorizontal * horizontalThresholdRatio);

	// Find vertical bounds (left and right)
	int left = 0, right = binary.cols - 1;
	for (int col = 0; col < binary.cols; ++col) {
		if (vertical_hist.at<int>(0, col) > verticalThreshold) {
			left = col;
			break;
		}
	}
	for (int col = binary.cols - 1; col >= 0; --col) {
		if (vertical_hist.at<int>(0, col) > verticalThreshold) {
			right = col;
			break;
		}
	}

	// Find horizontal bounds (top and bottom)
	int top = 0, bottom = binary.rows - 1;
	for (int row = 0; row < binary.rows; ++row) {
		if (horizontal_hist.at<int>(row, 0) > horizontalThreshold) {
			top = row;
			break;
		}
	}
	for (int row = binary.rows - 1; row >= 0; --row) {
		if (horizontal_hist.at<int>(row, 0) > horizontalThreshold) {
			bottom = row;
			break;
		}
	}

	// Crop the image
	if (left < right && top < bottom) {
		Rect roi(left, top, right - left + 1, bottom - top + 1);
		return binary(roi);
	}
	else {
		std::cerr << "Could not find suitable bounds. Returning original image." << std::endl;
		return binary.clone();
	}
}

Mat cutPlateRegion(const Mat& binary) {
	// Invert the image: license plates are white (high intensity) on black background
	Mat inverted;
	bitwise_not(binary, inverted);

	// Find contours of white regions
	std::vector<std::vector<Point>> contours;
	findContours(inverted, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Locate the largest contour (assume it's the plate)
	Rect boundingBox;
	int maxArea = 0;
	for (const auto& contour : contours) {
		Rect box = boundingRect(contour);
		int area = box.width * box.height;
		if (area > maxArea) {  // Keep the largest white region
			boundingBox = box;
			maxArea = area;
		}
	}

	// Crop to the bounding box
	if (maxArea > 0) {
		return binary(boundingBox);
	}
	else {
		std::cerr << "No valid region found. Returning original image." << std::endl;
		return binary.clone();
	}
}

Mat computeProjections(const Mat& binary_img) {
	int h = binary_img.rows;
	int w = binary_img.cols;
	std::vector<int> projectionHoriz(h, 0);
	std::vector<int> projectionVert(w, 0);
	Mat projection_img(h, w, CV_8UC1, Scalar(255));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (binary_img.at<uchar>(i, j) == 0) {
				projectionHoriz[i]++;
			}
		}
	}
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < h; i++) {
			if (binary_img.at<uchar>(i, j) == 0) {
				projectionVert[j]++;
			}
		}
	}
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < projectionVert[j]; i++) {
			projection_img.at<uchar>(i, j) = 0;
		}
	}
	return projection_img;
}

double percentageBlack(const Mat& src) {
	int total = src.cols * src.rows;
	double blackCount = 0.0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				blackCount = blackCount + 1.0;
			}
		}
	}
	if (show) {
		std::cout << "Black percentage: " << (double)(blackCount / total) << std::endl;
	}
	return (double)(blackCount / total);
}

std::vector<Mat> segmentCharactersUsingProj(const Mat& roi, const Mat& projection, double thresholdProj, double threasholdBlack) {
	int h = roi.rows;
	int w = roi.cols;

	int topMarginCut = 0;
	for (int i = 0; i < h; i++) {
		int blackPixels = 0;
		for (int j = 0; j < w; j++) {
			if (projection.at<uchar>(i, j) == 0) {
				blackPixels++;
			}
		}
		if (blackPixels < w * thresholdProj) {
			topMarginCut = i + 1;
			printf("Top Margin Cut %d with value of black pixels: %d\n", i, blackPixels);
			break;
		}
	}
	Mat projCropped = projection(Rect(0, topMarginCut, w, h - topMarginCut));
	/*if (show) {
		imshow("Projection Cropped", projCropped);
	}*/
	std::vector<int> projectionVert(w, 0);

	for (int j = 0; j < w; j++) {
		for (int i = 0; i < h - topMarginCut; i++) {
			if (projCropped.at<uchar>(i, j) == 0) {
				projectionVert[j]++;
			}
		}
	}

	std::vector<int> boundaries;
	bool inSegment = false;

	for (int j = 0; j < w; j++) {
		if (projectionVert[j] > 0 && !inSegment) {
			boundaries.push_back(j);
			inSegment = true;
		}
		else if (projectionVert[j] == 0 && inSegment) {
			boundaries.push_back(j - 1);
			inSegment = false;
		}
	}
	if (inSegment) {
		boundaries.push_back(w - 1);
	}

	std::vector<Mat> characters;
	for (size_t i = 0; i < boundaries.size(); i += 2) {
		int startCol = boundaries[i];
		int endCol = boundaries[i + 1];
		Rect charRect(startCol, topMarginCut, endCol - startCol + 1, h - topMarginCut);
		Mat character = roi(charRect).clone();
		printf("Percentage Black: %f\n", percentageBlack(character));
		if (threasholdBlack < percentageBlack(character) && percentageBlack(character) < 0.95) {
			characters.push_back(character);
		}
	}
	return characters;
}

int classifyBayes(Mat img, Mat priors, Mat likelihood) {
	img.convertTo(img, CV_64F);
	Mat flat = img.reshape(1, 1);
	double maxLogPosterior = -DBL_MAX;
	int bestClass = -1;
	std::vector<double> logPosteriors(priors.rows, 0.0);
	std::ofstream log_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/log_bayes.txt", std::ios::trunc);
	if (log_file.is_open()) {
		log_file << "New Run: Log Posterior Values and Probabilities\n";
		log_file << "--------------------------------------\n";
		log_file << "Flat dimensions: " << flat.size() << std::endl;
		log_file << "Likelihood dimensions: " << likelihood.size() << std::endl;
		log_file.close();
	}
	log_file.open("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/log_bayes.txt", std::ios::app);
	if (log_file.is_open()) {
		log_file << "Priors rows: " << priors.rows << std::endl;
		double epsilon = 1e-10;
		for (int c = 0; c < priors.rows; c++) {
			double logPosterior = log(priors.at<double>(c, 0));
			log_file << "Initial Log Posterior for class " << c << " = " << logPosterior << " for prior = " << priors.at<double>(c, 0) << std::endl;
			for (int j = 0; j < flat.cols; j++) {
				//log_file << "Likelihood at (" << c << ", " << j << ") = " << likelihood.at<double>(c, j) << std::endl;
				if (flat.at<double>(0, j) == 0) {
					logPosterior += log(1.0 - likelihood.at<double>(c, j) + epsilon);
				}
				else {
					logPosterior += log(likelihood.at<double>(c, j) + epsilon);
					//log_file << "Flat = " << flat.at<double>(0, j) << " for j = " << j << std::endl;
				}
			}
			logPosteriors[c] = logPosterior;
			log_file << "Log Posterior = " << logPosterior << std::endl;
			if (maxLogPosterior < logPosterior) {
				bestClass = c;
				maxLogPosterior = logPosterior;
			}
		}
		log_file << "Log posterior values for each class:" << std::endl;
		for (int c = 0; c < priors.rows; c++) {
			log_file << "Class " << c << ": " << logPosteriors[c] << std::endl;
		}
		double logSumExp = 0.0;
		for (double logPosterior : logPosteriors) {
			logSumExp += exp(logPosterior - maxLogPosterior);
		}
		logSumExp = maxLogPosterior + log(logSumExp);
		std::vector<double> probabilities(priors.rows, 0.0);
		for (int c = 0; c < priors.rows; c++) {
			probabilities[c] = exp(logPosteriors[c] - logSumExp);
		}
		log_file << "Probabilities for each class:" << std::endl;
		for (int c = 0; c < priors.rows; c++) {
			log_file << "Class " << c << ": " << probabilities[c] << std::endl;
		}
	}
	log_file.close();
	return bestClass;
}

class CustomBayesClassifier {
public:
	CustomBayesClassifier(cv::Mat priors, cv::Mat likelihood)
		: priors(priors), likelihood(likelihood) {
	}
	int predict(const cv::Mat& sample) const {
		return classifyBayes(sample, priors, likelihood);
	}
private:
	cv::Mat priors;
	cv::Mat likelihood;
};

void computeGradients(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& angle) {
	cv::Mat gx, gy;
	cv::Sobel(image, gx, CV_64F, 1, 0, 3);
	cv::Sobel(image, gy, CV_64F, 0, 1, 3);
	magnitude = cv::Mat(image.size(), CV_64F);
	angle = cv::Mat(image.size(), CV_64F);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			magnitude.at<double>(i, j) = std::sqrt(gx.at<double>(i, j) * gx.at<double>(i, j) +
				gy.at<double>(i, j) * gy.at<double>(i, j));
			angle.at<double>(i, j) = std::atan2(gy.at<double>(i, j), gx.at<double>(i, j)) * (180 / CV_PI);
			if (angle.at<double>(i, j) < 0) {
				angle.at<double>(i, j) += 180;
			}
		}
	}
}

std::vector<double> computeClassifierWeights(
	const std::vector<std::shared_ptr<cv::ml::StatModel>>& classifiers,
	const std::vector<std::shared_ptr<CustomBayesClassifier>>& customClassifiers,
	const std::vector<cv::Mat>& validationSamples,
	const std::vector<int>& validationLabels,
	double bayesWeightScale) {

	std::vector<double> weights(classifiers.size() + customClassifiers.size(), 0.0);
	std::vector<double> performances;

	for (size_t i = 0; i < classifiers.size(); ++i) {
		int correct = 0;
		for (size_t j = 0; j < validationSamples.size(); ++j) {
			int predicted = classifiers[i]->predict(validationSamples[j]);
			if (predicted == validationLabels[j]) {
				correct++;
			}
		}
		double accuracy = static_cast<double>(correct) / validationSamples.size();
		performances.push_back(accuracy);
	}

	for (size_t i = 0; i < customClassifiers.size(); ++i) {
		int correct = 0;
		for (size_t j = 0; j < validationSamples.size(); ++j) {
			int predicted = customClassifiers[i]->predict(validationSamples[j]);
			if (predicted == validationLabels[j]) {
				correct++;
			}
		}
		double accuracy = static_cast<double>(correct) / validationSamples.size();
		performances.push_back(accuracy * bayesWeightScale);
	}

	double sumPerformance = std::accumulate(performances.begin(), performances.end(), 0.0);
	for (size_t i = 0; i < performances.size(); ++i) {
		weights[i] = performances[i] / sumPerformance;
	}

	return weights;
}

void computeHistograms(const cv::Mat& magnitude, const cv::Mat& angle, int nbins, int cell_size,
	std::vector<std::vector<double>>& histograms) {
	for (int i = 0; i < magnitude.rows; i += cell_size) {
		for (int j = 0; j < magnitude.cols; j += cell_size) {
			std::vector<double> cell_hist(nbins, 0);

			for (int y = i; y < i + cell_size && y < magnitude.rows; y++) {
				for (int x = j; x < j + cell_size && x < magnitude.cols; x++) {
					int bin_idx = static_cast<int>(angle.at<double>(y, x) / (180 / nbins));
					bin_idx = min(max(bin_idx, 0), nbins - 1);
					cell_hist[bin_idx] += magnitude.at<double>(y, x);
				}
			}
			histograms.push_back(cell_hist);
		}
	}
}

void computeHOG(const cv::Mat& image, int cell_size, int block_size, int nbins, std::vector<double>& hog_features) {
	if (show) {
		imshow("Input", image);
	}
	cv::Mat image_resized;
	cv::resize(image, image_resized, cv::Size(128, 64));
	if (show) {
		imshow("Resized", image);
	}
	cv::Mat magnitude, angle;
	computeGradients(image_resized, magnitude, angle);
	if (show) {
		imshow("Magnitude", magnitude);
		imshow("Angle", angle);
	}
	std::vector<std::vector<double>> cell_histograms;
	computeHistograms(magnitude, angle, nbins, cell_size, cell_histograms);
	for (size_t i = 0; i < cell_histograms.size() - (block_size - 1) * cell_size; i++) {
		std::vector<double> block_hist;
		for (int b = 0; b < block_size * block_size; b++) {
			block_hist.insert(block_hist.end(), cell_histograms[i + b].begin(), cell_histograms[i + b].end());
		}
		double norm_factor = 0.0;
		for (double value : block_hist) {
			norm_factor += value * value;
		}
		norm_factor = std::sqrt(norm_factor + 1e-6);

		for (double& value : block_hist) {
			value /= norm_factor;
		}
		hog_features.insert(hog_features.end(), block_hist.begin(), block_hist.end());
	}
	waitKey();
}

Mat invertedBW(const Mat& src) {
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

Mat binaryThreshold(Mat src) {
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) < 128) {
				dst.at<uchar>(i, j) = 0;
			}
			else {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	return dst;
}

void twoPassComponentLabelingNew(const Mat& src, std::vector<Rect>& boundingBoxes) {
	Mat labels(src.rows, src.cols, CV_32SC1, Scalar(0));
	int h = src.rows;
	int w = src.cols;
	int label = 0;
	std::vector<std::vector<int>> edges(10000);
	Mat inverted = invertedBW(src);

	// First pass: Assign labels and build equivalence graph
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (inverted.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				std::vector<int> neighbors;

				if (i > 0 && j > 0 && labels.at<int>(i - 1, j - 1) > 0) neighbors.push_back(labels.at<int>(i - 1, j - 1));
				if (i > 0 && labels.at<int>(i - 1, j) > 0) neighbors.push_back(labels.at<int>(i - 1, j));
				if (i > 0 && j + 1 < w && labels.at<int>(i - 1, j + 1) > 0) neighbors.push_back(labels.at<int>(i - 1, j + 1));
				if (j > 0 && labels.at<int>(i, j - 1) > 0) neighbors.push_back(labels.at<int>(i, j - 1));

				if (neighbors.empty()) {
					label++;
					labels.at<int>(i, j) = label;
				}
				else {
					int minLabel = *min_element(neighbors.begin(), neighbors.end());
					labels.at<int>(i, j) = minLabel;

					for (int n : neighbors) {
						if (n != minLabel) {
							edges[minLabel].push_back(n);
							edges[n].push_back(minLabel);
						}
					}
				}
			}
		}
	}

	// Resolve label equivalences using BFS
	std::vector<int> newLabels(label + 1, 0);
	int newLabel = 0;
	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int> q;
			newLabels[i] = newLabel;
			q.push(i);

			while (!q.empty()) {
				int current = q.front();
				q.pop();
				for (int neighbor : edges[current]) {
					if (newLabels[neighbor] == 0) {
						newLabels[neighbor] = newLabel;
						q.push(neighbor);
					}
				}
			}
		}
	}

	// Second pass: Collect bounding boxes
	boundingBoxes.clear();
	std::vector<std::vector<Point>> contours(newLabel);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int lbl = labels.at<int>(i, j);
			if (lbl > 0) {
				int finalLabel = newLabels[lbl] - 1;
				contours[finalLabel].push_back(Point(j, i));
			}
		}
	}

	// Generate bounding rectangles for each labeled region
	for (const auto& contour : contours) {
		if (!contour.empty()) {
			boundingBoxes.push_back(boundingRect(contour));
		}
	}
}

double edgeDensity(const Mat& src) {
	Mat edges;
	Canny(src, edges, 50, 150); // Apply Canny edge detection

	int edgeCount = countNonZero(edges); // Count edge pixels
	int totalPixels = src.cols * src.rows;
	if (show) {
		std::cout << "Edge Density: " << (double)edgeCount / totalPixels << std::endl;
	}
	return (double)edgeCount / totalPixels;
}

bool isValidAspectRatio(const Rect& box) {
	double aspectRatio = (double)box.width / box.height;
	if (show) {
		std::cout << "Aspect ratio: " << (double)aspectRatio << std::endl;
	}
	return (aspectRatio >= 2.0 && aspectRatio <= 5.0);
}

bool isLandscapeRectangle(const Rect& box) {
	if (box.width > box.height) {
		return true;
	}
	else {
		return false;
	}
}

std::vector<Mat> findLicensePlateCandidates(const Mat& src, const std::vector<Rect>& boundingBoxes) {
	std::vector<Mat> candidates;
	int index = 1;
	for (const auto& box : boundingBoxes) {
		//std::cout << "Candidate " << index << std::endl;
		// Extract region of interest (ROI)
		Mat roi = src(box);

		// Check black pixel percentage
		double blackPercent = percentageBlack(roi);
		if (blackPercent < 0.3 || blackPercent > 0.6) {
			continue; // Reject if black percentage is not in range
		}

		// Check edge density
		double density = edgeDensity(roi);
		if (density < 0.1) {
			continue; // Reject if edge density is too low
		}

		// Check aspect ratio
		if (!isValidAspectRatio(box)) {
			continue; // Reject if aspect ratio is invalid
		}

		if (roi.rows != src.rows || roi.cols != src.cols) {
			if (!isLandscapeRectangle(box)) {
				continue;
			}
		}

		// If all checks pass, save the candidate
		candidates.push_back(roi);
		//std::cout << "Candidate " << index << " accepted" << std::endl;
		index++;
	}

	return candidates;
}

std::vector<int> weightedVotingClassifier(
	const std::vector<std::shared_ptr<cv::ml::StatModel>>& classifiers,
	const std::vector<std::shared_ptr<CustomBayesClassifier>>& customClassifiers,
	const std::vector<double>& classifierWeights,
	const std::vector<double>& customClassifierWeights,
	const std::vector<cv::Mat>& samples) {

	assert(classifiers.size() == classifierWeights.size());
	assert(customClassifiers.size() == customClassifierWeights.size());

	std::vector<int> predictions;

	for (const auto& sample : samples) {
		std::map<int, double> classVotes;

		for (size_t i = 0; i < classifiers.size(); ++i) {
			int prediction = classifiers[i]->predict(sample);
			classVotes[prediction] += classifierWeights[i];
		}

		for (size_t i = 0; i < customClassifiers.size(); ++i) {
			int prediction = customClassifiers[i]->predict(sample);
			classVotes[prediction] += customClassifierWeights[i];
		}

		auto maxElement = std::max_element(classVotes.begin(), classVotes.end(),
			[](const auto& a, const auto& b) { return a.second < b.second; });

		predictions.push_back(maxElement->first);
	}

	return predictions;
}

std::string getFilenameFromPath(const std::string& filepath) {
	size_t pos = filepath.find_last_of("/");
	if (pos != std::string::npos) {
		return filepath.substr(pos + 1);
	}
	else {
		return filepath;
	}
}

std::string cleanText(const std::string& text) {
	std::string cleanedText = text;
	cleanedText.erase(std::remove_if(cleanedText.begin(), cleanedText.end(),
		[](unsigned char c) { return std::isspace(c); }),
		cleanedText.end());

	return cleanedText;
}

void writeResultsToCSV(const std::string& filename, const std::vector<std::pair<std::string, std::string>>& dataset) {
	std::ofstream outputFile(filename, std::ios::app);

	if (!outputFile.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}
	for (size_t i = 0; i < dataset.size(); ++i) {
		std::string cleanedOutput = cleanText(dataset[i].second);
		outputFile << getFilenameFromPath(dataset[i].first) << ";" << cleanedOutput << std::endl;
	}
	outputFile.close();
}

std::vector<std::string> listFiles(const std::string& directoryPath) {
	std::vector<std::string> files;
	WIN32_FIND_DATAA findFileData;
	HANDLE hFind = FindFirstFileA((directoryPath + "/*").c_str(), &findFileData);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				files.push_back(directoryPath + "/" + findFileData.cFileName);
			}
		} while (FindNextFileA(hFind, &findFileData) != 0);
		FindClose(hFind);
	}
	return files;
}

void writeResultsToCSV5Columns(const std::string& filename, const std::vector<std::tuple<std::string, int, int, int, int>>& dataset) {
	std::ofstream outputFile(filename, std::ios::app);
	if (!outputFile.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}
	for (const auto& data : dataset) {
		outputFile << std::get<0>(data) << ";"
			<< std::get<1>(data) << ";"
			<< std::get<2>(data) << ";"
			<< std::get<3>(data) << ";"
			<< std::get<4>(data) << std::endl;
	}
	outputFile.close();
}

struct CandidateResult {
	std::string plateName;  // The plate name
	std::string candidateName;  // The candidate folder name
	int predictedLabel;  // The prediction result for this candidate
	int actualLabel;  // The actual label of the plate (for comparison)
};

struct TestImageInfo {
	int testIndex;
	std::string plateName;
	int candidateIndex;
	std::string imageName;
};

int levenshteinDistance(const std::string& s1, const std::string& s2) {
	int m = s1.size();
	int n = s2.size();
	std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

	for (int i = 0; i <= m; ++i) dp[i][0] = i;
	for (int j = 0; j <= n; ++j) dp[0][j] = j;

	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (s1[i - 1] == s2[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = 1 + min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
			}
		}
	}
	return dp[m][n];
}

int countCommonCharacters(const std::string& s1, const std::string& s2) {
	std::string s1Copy = s1, s2Copy = s2;
	std::sort(s1Copy.begin(), s1Copy.end());
	std::sort(s2Copy.begin(), s2Copy.end());

	int count = 0;
	auto it1 = s1Copy.begin();
	auto it2 = s2Copy.begin();

	while (it1 != s1Copy.end() && it2 != s2Copy.end()) {
		if (*it1 == *it2) {
			++count;
			++it1;
			++it2;
		}
		else if (*it1 < *it2) {
			++it1;
		}
		else {
			++it2;
		}
	}
	return count;
}

void updateConfusionMatrix(const std::string& actual, const std::string& predicted,
	std::unordered_map<char, std::unordered_map<char, int>>& confusionMatrix) {
	int actualLen = actual.size();
	int predictedLen = predicted.size();
	int maxLen = max(actualLen, predictedLen);

	for (int i = 0; i < maxLen; ++i) {
		char actualChar = i < actualLen ? actual[i] : '-'; // Use '-' for extra characters in `predicted`
		char predictedChar = i < predictedLen ? predicted[i] : '-'; // Use '-' for extra characters in `actual`

		confusionMatrix[actualChar][predictedChar]++;
	}
}

void printConfusionMatrix(const std::unordered_map<char, std::unordered_map<char, int>>& confusionMatrix,
	std::ostream& result_file) {
	// Collect all unique characters (actual and predicted)
	std::set<char> uniqueChars;
	for (const auto& [actualChar, predictions] : confusionMatrix) {
		uniqueChars.insert(actualChar);
		for (const auto& [predictedChar, _] : predictions) {
			uniqueChars.insert(predictedChar);
		}
	}

	// Calculate the maximum width for alignment
	int maxWidth = 1; // Minimum width for single characters
	for (char c : uniqueChars) {
		maxWidth = max(maxWidth, (int)std::to_string(c).length());
	}
	for (const auto& [actualChar, predictions] : confusionMatrix) {
		for (const auto& [predictedChar, count] : predictions) {
			maxWidth = max(maxWidth, (int)std::to_string(count).length());
		}
	}

	// Adjust width for a clean table
	auto pad = [&](const std::string& str) {
		return std::string(maxWidth - str.length(), ' ') + str;
		};

	// Print the header row
	result_file << std::string(maxWidth, ' ') << " ";
	for (char predictedChar : uniqueChars) {
		result_file << pad(std::string(1, predictedChar)) << " ";
	}
	result_file << std::endl;

	// Print the matrix rows
	for (char actualChar : uniqueChars) {
		result_file << pad(std::string(1, actualChar)) << " ";
		for (char predictedChar : uniqueChars) {
			int count = confusionMatrix.count(actualChar) && confusionMatrix.at(actualChar).count(predictedChar)
				? confusionMatrix.at(actualChar).at(predictedChar)
				: 0;
			result_file << pad(std::to_string(count)) << " ";
		}
		result_file << std::endl;
	}
}

void printMetrics(const std::unordered_map<char, std::unordered_map<char, int>>& confusionMatrix, std::ostream& result_file) {
	int TP = 0, TN = 0, FP = 0, FN = 0;
	int totalPositive = 0, totalNegative = 0;

	// Calculate TP, TN, FP, FN for each character pair in confusion matrix
	for (const auto& [actualChar, predictions] : confusionMatrix) {
		for (const auto& [predictedChar, count] : predictions) {
			if (actualChar == predictedChar) {
				TP += count; // True Positive
			}
			else {
				FN += count; // False Negative for actual
				FP += count; // False Positive for predicted
			}
		}
	}

	// Accuracy
	int total = TP + FP + FN;
	double accuracy = (total > 0) ? (double)(TP) / total : 0;

	// Precision
	double precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0;

	// Recall
	double recall = (TP + FN > 0) ? (double)TP / (TP + FN) : 0;

	// F1 score
	double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

	// Print Metrics
	result_file << "Accuracy: " << accuracy << std::endl;
	result_file << "Precision: " << precision << std::endl;
	result_file << "Recall: " << recall << std::endl;
	result_file << "F1 Score: " << f1 << std::endl;
}

void analyzePredictions(const std::unordered_map<std::string, std::unordered_map<int, std::string>>& platePredictions,
	std::ostream& result_file) {
	int totalLevenshtein = 0;
	int totalCommonChars = 0;
	int totalLengthDiff = 0;
	int totalCandidates = 0;

	int bestTotalLevenshtein = 0;
	int bestTotalCommonChars = 0;
	int bestTotalLengthDiff = 0;
	int totalPlates = 0;
	int totalPlatesLength = 0;

	int zeroLengthDiffCount = 0; // Counter for candidates with zero length difference
	int zeroLengthDiffLevenshtein = 0; // Sum of Levenshtein distances for these candidates
	int zeroLengthDiffCommonChars = 0; // Sum of common characters for these candidates

	// Confusion matrix to track character-level misclassifications
	std::unordered_map<char, std::unordered_map<char, int>> confusionMatrix;

	for (const auto& [plateName, candidates] : platePredictions) {
		result_file << "Analysis for Plate: " << plateName << std::endl;

		int bestLevenshtein = INT_MAX;
		int bestCommonChars = 0;
		int bestLengthDiff = INT_MAX;
		std::string bestCandidate;

		// Find the best candidate for the current plate
		for (const auto& [candidateIndex, mergedString] : candidates) {
			int levenshtein = levenshteinDistance(plateName, mergedString);
			int commonChars = countCommonCharacters(plateName, mergedString);
			int lengthDiff = std::abs((int)plateName.size() - (int)mergedString.size());

			// Track metrics for candidates with zero length difference
			if (lengthDiff == 0) {
				zeroLengthDiffCount++;
				zeroLengthDiffLevenshtein += levenshtein;
				zeroLengthDiffCommonChars += commonChars;
			}

			// Update the best candidate based on the new rules
			if (levenshtein < bestLevenshtein ||
				(levenshtein == bestLevenshtein && commonChars > bestCommonChars) ||
				(levenshtein == bestLevenshtein && commonChars == bestCommonChars && lengthDiff < bestLengthDiff)) {
				bestLevenshtein = levenshtein;
				bestCommonChars = commonChars;
				bestLengthDiff = lengthDiff;
				bestCandidate = mergedString;
			}
		}

		// Update confusion matrix and metrics for the best candidate only
		updateConfusionMatrix(plateName, bestCandidate, confusionMatrix);

		// Update overall analysis metrics for the best candidate
		bestTotalLevenshtein += bestLevenshtein;
		bestTotalCommonChars += bestCommonChars;
		bestTotalLengthDiff += bestLengthDiff;
		++totalPlates;
		totalPlatesLength += (int)plateName.size();

		// Print results for the best candidate
		result_file << "  Best Candidate: " << bestCandidate << std::endl;
		result_file << "    Levenshtein Distance: " << bestLevenshtein << std::endl;
		result_file << "    Common Characters: " << bestCommonChars << std::endl;
		result_file << "    Length Difference: " << bestLengthDiff << std::endl;
	}

	// Overall analysis for best candidates only
	result_file << "Overall Analysis (Best Candidates):" << std::endl;
	result_file << "  Average Levenshtein Distance: " << (totalPlates > 0 ? bestTotalLevenshtein / totalPlates : 0) << std::endl;
	result_file << "  Total Common Characters: " << bestTotalCommonChars << std::endl;
	result_file << "  Average Length Difference: " << (totalPlates > 0 ? bestTotalLengthDiff / totalPlates : 0) << std::endl;

	// Print zero length difference metrics
	result_file << "Metrics for Candidates with Zero Length Difference:" << std::endl;
	result_file << "  Number of Candidates: " << zeroLengthDiffCount << std::endl;
	result_file << "  Average Levenshtein Distance: "
		<< (zeroLengthDiffCount > 0 ? zeroLengthDiffLevenshtein / zeroLengthDiffCount : 0) << std::endl;
	result_file << "  Average Common Characters: "
		<< (zeroLengthDiffCount > 0 ? zeroLengthDiffCommonChars / zeroLengthDiffCount : 0) << std::endl;

	// Print confusion matrix and metrics for the best candidates
	printConfusionMatrix(confusionMatrix, result_file);
	printMetrics(confusionMatrix, result_file);
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		//Project
		printf(" 1 - New from segmentation method, for PRS\n");
		printf(" 2 - Process all images in a folder\n");
		printf(" 3 - HOG\n");
		printf(" 4 - Naive Bayes on Characters Dataset\n");
		printf(" 5 - Naive Bayes and HOG on Characters Dataset\n");
		printf(" 6 - Naive Bayes and HOG on Single Image\n");
		printf(" 7 - Weighted Voting mechanism on Characters Dataset\n");
		printf(" 8 - HOG + Naive Bayes + Weighted Voting mechanism on Resulted Characters Dataset\n");
		printf(" 9 - Naive Bayes + Weighted Voting mechanism on Resulted Characters Dataset\n");
		printf(" 10 - Naive Bayes + Weighted Voting mechanism on Partial Resulted Characters Dataset\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
		{
			char fname[MAX_PATH];
			double percentageV, percentageH, percentageG, percentageB, percentageCh = 0.0;
			int d, v, h, g, b, ch;
			printf("Give number of dilations: ");
			getchar();
			scanf("%d", &d);
			printf("Give percentage of white for vertically cut of borders: ");
			getchar();
			scanf("%d", &v);
			printf("Give percentage of white for horizontally cut of borders: ");
			getchar();
			scanf("%d", &h);
			printf("Give percentage for black pixels threshold in projection: ");
			getchar();
			scanf("%d", &b);
			printf("Give percentage for black pixels threshold in character image: ");
			getchar();
			scanf("%d", &ch);
			percentageV = (double)v / 100.0;
			percentageH = (double)h / 100.0;
			percentageB = (double)b / 100.0;
			percentageCh = (double)ch / 100.0;
			while (openFileDlg(fname))
			{
				std::string filePath(fname);
				std::string basePath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/charactersResulted";

				std::string folderName = filePath.substr(filePath.find_last_of("\\") + 1);
				folderName = folderName.substr(0, folderName.find_last_of('.'));

				std::string folderPath = basePath + "/" + folderName;

				if (fs::exists(folderPath)) {
					for (const auto& entry : fs::directory_iterator(folderPath)) {
						if (fs::is_directory(entry)) {
							fs::remove_all(entry); // Remove the entire subfolder
						}
					}
				}
				else {
					fs::create_directory(folderPath); // Create the folder if it doesn't exist
				}

				show = true;
				Mat initialImage = imread(fname);
				imshow("Input Image", initialImage);
				Mat resizedImage;
				resizeImg(initialImage, resizedImage, 750, true);
				imshow("Resized Image", resizedImage);

				Mat grayImage;
				RGBToGrayscale(resizedImage, grayImage);
				imshow("Gray Image", grayImage);

				Mat closedImage = closingGrayscale(grayImage);
				imshow("Closed Image", closedImage);

				Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
				imshow("Bilateral Filtered Image", bilateralFilteredImage);

				double k = 0.4;
				int pH = 50;
				int pL = static_cast<int>(k * pH);
				Mat gauss, cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				imshow("Gauss", gauss);

				Canny(gauss, cannyImage, pL, pH, 3);
				imshow("Canny", cannyImage);

				Mat cannyNegative = negativeTransform(cannyImage);
				imshow("Negative Canny", cannyNegative);

				Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 3);
				imshow("Dilated Canny", dilatedCanny);

				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat licensePlateImage = detectedPlate(plateRect);

				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}

				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);
				Mat grayLicensePlate;
				RGBToGrayscale(licensePlateImage, grayLicensePlate);
				imshow("Gray License", grayLicensePlate);
				Mat thLicense = basicGlobalThresholding(grayLicensePlate);
				imshow("Global Thresholding", thLicense);
				Mat dilatedPlate = repeatDilationVertical(thLicense, d);
				imshow("Dilated Plate", dilatedPlate);

				Mat roi = cutBorders(dilatedPlate, percentageV, percentageH);
				//Mat roi = cutBordersDynamic(dilatedPlate, percentageV, percentageH);
				imshow("Cut Borders", roi);

				std::vector<Rect> boundingBoxes;
				twoPassComponentLabelingNew(roi, boundingBoxes);

				// Draw bounding boxes on the image
				Mat boxed;
				cvtColor(roi, boxed, COLOR_GRAY2BGR);
				for (const auto& box : boundingBoxes) {
					rectangle(boxed, box, Scalar(0, 255, 0), 2);
				}

				// Show the result
				imshow("Labeled Components with Bounding Boxes", boxed);

				// Find license plate candidates
				std::vector<Mat> candidates = findLicensePlateCandidates(roi, boundingBoxes);

				if (candidates.size() == 0) {
					printf("No license plate candidates found. Adding the entire ROI as a candidate.\n");
					candidates.push_back(roi); // Add the ROI to candidates
				}

				// Display candidates
				for (size_t i = 0; i < candidates.size(); i++) {
					std::string windowName = "Candidate " + std::to_string(i);
					imshow(windowName, candidates[i]);
				}

				// Display candidates and process each one
				for (size_t i = 0; i < candidates.size(); i++) {
					// Create a window to display each candidate
					std::string windowName = "Candidate " + std::to_string(i);
					imshow(windowName, candidates[i]);

					// Create a subfolder for the current candidate
					std::string candidateFolderPath = folderPath + "/candidate" + std::to_string(i);
					if (!fs::exists(candidateFolderPath)) {
						fs::create_directory(candidateFolderPath);
					}

					// Compute projections and segment characters for the current candidate
					//candidates[i] = invertedBW(candidates[i]);
					Mat projection = computeProjections(candidates[i]);
					imshow("Projection for Candidate " + std::to_string(i), projection);

					// Segment characters based on the given thresholds
					std::vector<Mat> characters = segmentCharactersUsingProj(candidates[i], projection, percentageB, percentageCh);
					printf("Candidate %d - Characters found: %d\n", static_cast<int>(i), static_cast<int>(characters.size()));

					// Save each segmented character into the candidate subfolder
					for (size_t j = 0; j < characters.size(); ++j) {
						std::string characterFilePath = candidateFolderPath + "/" + std::to_string(j) + ".png";
						imwrite(characterFilePath, characters[j]);
						imshow("Candidate " + std::to_string(i) + " - Character " + std::to_string(j), characters[j]);
					}
				}

				waitKey();
			}
		}
		break;
		case 2:
		{
			std::string folderPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/dataset/imagesForTesting/plates";
			double percentageV, percentageH, percentageG, percentageB, percentageCh = 0.0;
			int d, v, h, g, b, ch;
			printf("Give number of dilations: ");
			getchar();
			scanf("%d", &d);
			printf("Give percentage of white for vertically cut of borders: ");
			getchar();
			scanf("%d", &v);
			printf("Give percentage of white for horizontally cut of borders: ");
			getchar();
			scanf("%d", &h);
			printf("Give percentage for black pixels threshold: ");
			getchar();
			scanf("%d", &b);
			printf("Give percentage for black pixels threshold in character image: ");
			getchar();
			scanf("%d", &ch);
			percentageV = (double)v / 100.0;
			percentageH = (double)h / 100.0;
			percentageB = (double)b / 100.0;
			percentageCh = (double)ch / 100.0;
			for (const auto& entry : fs::directory_iterator(folderPath))
			{
				if (entry.is_regular_file() && (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg"))
				{
					std::string fname = entry.path().string();

					std::string basePath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/charactersResulted";

					std::string fileName = entry.path().filename().string();
					std::string folderName = fileName.substr(0, fileName.find_last_of('.'));

					std::string folderToSave = basePath + "/" + folderName;
					if (!fs::exists(folderToSave))
					{
						fs::create_directory(folderToSave);
					}
					show = false;
					Mat initialImage = imread(fname);
					Mat resizedImage;
					resizeImg(initialImage, resizedImage, 750, true);
					Mat grayImage;
					RGBToGrayscale(resizedImage, grayImage);

					Mat closedImage = closingGrayscale(grayImage);

					Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);

					double k = 0.4;
					int pH = 50;
					int pL = static_cast<int>(k * pH);
					Mat gauss, cannyImage;
					GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);

					Canny(gauss, cannyImage, pL, pH, 3);

					Mat cannyNegative = negativeTransform(cannyImage);

					Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 3);

					Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
					Mat detectedPlate = resizedImage.clone();
					Mat licensePlateImage = detectedPlate(plateRect);

					if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
						licensePlateImage = initialImage.clone();
					}

					rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
					Mat grayLicensePlate;
					RGBToGrayscale(licensePlateImage, grayLicensePlate);
					Mat thLicense = basicGlobalThresholding(grayLicensePlate);
					Mat dilatedPlate = repeatDilationVertical(thLicense, d);

					Mat roi = cutBorders(dilatedPlate, percentageV, percentageH);
					std::vector<Rect> boundingBoxes;
					twoPassComponentLabelingNew(roi, boundingBoxes);

					std::vector<Mat> candidates = findLicensePlateCandidates(roi, boundingBoxes);

					if (candidates.size() == 0) {
						printf("No license plate candidates found. Adding the entire ROI as a candidate.\n");
						candidates.push_back(roi);
					}

					for (size_t i = 0; i < candidates.size(); i++) {

						std::string candidateFolderPath = folderToSave + "/candidate" + std::to_string(i);
						if (!fs::exists(candidateFolderPath)) {
							fs::create_directory(candidateFolderPath);
						}

						Mat projection = computeProjections(candidates[i]);
						std::vector<Mat> characters = segmentCharactersUsingProj(candidates[i], projection, percentageB, percentageCh);
						for (size_t j = 0; j < characters.size(); ++j) {
							std::string characterFilePath = candidateFolderPath + "/" + std::to_string(j) + ".png";
							imwrite(characterFilePath, characters[j]);
						}
					}
					waitKey();
				}
			}
		}
		break;
		case 3:
		{
			char fname[MAX_PATH];
			while (openFileDlg(fname))
			{
				Mat image = imread(fname, cv::IMREAD_GRAYSCALE);
				show = true;
				if (image.empty()) {
					std::cerr << "Error: Unable to load the image." << std::endl;
					return -1;
				}
				int cell_size = 8;
				int block_size = 2;
				int nbins = 9;

				std::vector<double> hog_features;
				imshow("Input", image);
				Mat binarized = basicGlobalThresholding(image);
				imshow("Binarized", binarized);
				Mat invertedImage = invertedBW(binarized);
				computeHOG(image, cell_size, block_size, nbins, hog_features);
				std::cout << "Computed HOG Features size: " << hog_features.size() << std::endl;
				std::cout << "Computed HOG Features: ";
				for (int i = 0; i < hog_features.size(); i++) {
					std::cout << hog_features[i] << " ";
				}
				std::cout << std::endl;
				waitKey();
			}
		}
		break;
		case 4:
		{
			{
				std::ofstream result_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/evaluation_results_characters.txt", std::ios::app);
				if (result_file.is_open()) {
					result_file << "\n--------------------------------------\n";
					const int C = 36;
					const int d = 64 * 64;
					std::vector<Mat> trainImages[C], testImages[C];
					Mat X, y, X_test, y_test;
					srand(static_cast<unsigned int>(time(0)));

					for (int c = 0; c < C; ++c) {
						std::string folderName;
						std::string prefix;

						if (c < 10) {
							folderName = "digit_" + std::to_string(c);
							prefix = std::to_string(c);
						}
						else {
							folderName = "letter_" + std::string(1, char('A' + (c - 10)));
							prefix = std::string(1, char('A' + (c - 10)));
						}

						std::string folderPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/characters/" + folderName;
						std::vector<std::string> filenames;
						for (int index = 1; ; ++index) {
							char fname[256];
							sprintf(fname, "%s/%s_%d.png", folderPath.c_str(), prefix.c_str(), index);

							Mat img = imread(fname, 0);
							if (img.empty()) break;

							filenames.push_back(std::string(fname));
						}

						std::random_device rd;
						std::mt19937 g(rd());
						std::shuffle(filenames.begin(), filenames.end(), g);

						int numTrain = static_cast<int>(filenames.size() * 0.8);
						int numTest = filenames.size() - numTrain;

						for (int i = 0; i < numTrain; ++i) {
							Mat img = imread(filenames[i], 0);
							Mat resized;
							cv::resize(img, resized, cv::Size(64, 64));
							trainImages[c].push_back(resized);
						}

						for (int i = numTrain; i < filenames.size(); ++i) {
							Mat img = imread(filenames[i], 0);
							Mat resized;
							cv::resize(img, resized, cv::Size(64, 64));
							testImages[c].push_back(resized);
						}
					}

					int totalTrainSamples = 0, totalTestSamples = 0;
					for (int c = 0; c < C; ++c) {
						totalTrainSamples += trainImages[c].size();
						totalTestSamples += testImages[c].size();
					}

					X = Mat(totalTrainSamples, d, CV_64FC1);
					y = Mat(totalTrainSamples, 1, CV_64FC1);
					X_test = Mat(totalTestSamples, d, CV_64FC1);
					y_test = Mat(totalTestSamples, 1, CV_64FC1);
					int trainIndex = 0;
					for (int c = 0; c < C; ++c) {
						for (size_t i = 0; i < trainImages[c].size(); ++i) {
							Mat binarized = basicGlobalThresholding(trainImages[c][i]);
							Mat inverted = invertedBW(binarized);
							Mat flat = inverted.reshape(1, 1);
							flat.convertTo(flat, CV_64FC1);

							flat.copyTo(X.row(trainIndex));
							y.at<double>(trainIndex, 0) = c;
							trainIndex++;
						}
					}

					int testIndex = 0;
					for (int c = 0; c < C; ++c) {
						for (size_t i = 0; i < testImages[c].size(); ++i) {
							Mat binarized = basicGlobalThresholding(testImages[c][i]);
							Mat inverted = invertedBW(binarized);
							Mat flat = inverted.reshape(1, 1);
							flat.convertTo(flat, CV_64FC1);

							flat.copyTo(X_test.row(testIndex));
							y_test.at<double>(testIndex, 0) = c;
							testIndex++;
						}
					}

					Mat priors(C, 1, CV_64FC1);
					Mat likelihood(C, d, CV_64FC1, Scalar(1));
					for (int c = 0; c < C; ++c) {
						Mat classSamples = X.rowRange(trainIndex * c / C, trainIndex * (c + 1) / C);
						int classCount = classSamples.rows;

						priors.at<double>(c, 0) = static_cast<double>(classCount) / totalTrainSamples;

						for (int j = 0; j < d; ++j) {
							double count = 0.0;
							for (int k = 0; k < classSamples.rows; ++k) {
								count += (classSamples.at<double>(k, j) == 255 ? 1.0 : 0.0);
							}
							if (count == 0) {
								likelihood.at<double>(c, j) = (count + 1) / (classCount + C);
							}
							else {
								likelihood.at<double>(c, j) = count / classCount;
							}
						}

					}

					int correct = 0, total = 0;
					Mat confusionMatrix = Mat::zeros(C, C, CV_32S);
					for (int i = 0; i < X_test.rows; ++i) {
						Mat img = X_test.row(i).reshape(1, 64);
						int trueClass = static_cast<int>(y_test.at<double>(i, 0));
						int predictedClass = classifyBayes(img, priors, likelihood);
						confusionMatrix.at<int>(predictedClass, trueClass)++;
						if (trueClass == predictedClass) correct++;
						total++;
					}

					double errorRate = 1.0 - (double)correct / total;

					result_file << "Train size: " << totalTrainSamples << std::endl;
					result_file << "Test size: " << totalTestSamples << std::endl;
					result_file << "Error Rate: " << errorRate << std::endl;
					result_file << "Confusion Matrix: " << std::endl << confusionMatrix << std::endl;
				}
				result_file.close();
			}

		}
		break;
		case 5:
		{
			std::ofstream result_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/evaluation_results_characters_hog.txt", std::ios::app);
			if (result_file.is_open()) {
				result_file << "\n--------------------------------------\n";
				const int C = 36;
				const int d = 128 * 64;
				show = false;
				std::vector<Mat> trainImages[C], testImages[C];
				Mat X, y, X_test, y_test;
				srand(static_cast<unsigned int>(time(0)));
				for (int c = 0; c < C; ++c) {
					std::string folderName;
					std::string prefix;

					if (c < 10) {
						folderName = "digit_" + std::to_string(c);
						prefix = std::to_string(c);
					}
					else {
						folderName = "letter_" + std::string(1, char('A' + (c - 10)));
						prefix = std::string(1, char('A' + (c - 10)));
					}

					std::string folderPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/characters/" + folderName;
					std::vector<std::string> filenames;
					for (int index = 1; ; ++index) {
						char fname[256];
						sprintf(fname, "%s/%s_%d.png", folderPath.c_str(), prefix.c_str(), index);
						Mat img = imread(fname, 0);
						if (img.empty()) break;
						filenames.push_back(std::string(fname));
					}

					std::random_device rd;
					std::mt19937 g(rd());
					std::shuffle(filenames.begin(), filenames.end(), g);

					int numTrain = static_cast<int>(filenames.size() * 0.8);
					int numTest = filenames.size() - numTrain;

					for (int i = 0; i < numTrain; ++i) {
						Mat img = imread(filenames[i], 0);
						Mat resized;
						cv::resize(img, resized, cv::Size(128, 64));
						Mat binarized = basicGlobalThresholding(resized);
						trainImages[c].push_back(binarized);
					}

					for (int i = numTrain; i < filenames.size(); ++i) {
						Mat img = imread(filenames[i], 0);
						Mat resized;
						cv::resize(img, resized, cv::Size(128, 64));
						Mat binarized = basicGlobalThresholding(resized);
						testImages[c].push_back(binarized);
					}
				}

				int totalTrainSamples = 0, totalTestSamples = 0;
				for (int c = 0; c < C; ++c) {
					totalTrainSamples += trainImages[c].size();
					totalTestSamples += testImages[c].size();
				}

				X = Mat(totalTrainSamples, 4320, CV_64FC1);
				y = Mat(totalTrainSamples, 1, CV_64FC1);
				X_test = Mat(totalTestSamples, 4320, CV_64FC1);
				y_test = Mat(totalTestSamples, 1, CV_64FC1);

				int trainIndex = 0;
				for (int c = 0; c < C; ++c) {
					for (size_t i = 0; i < trainImages[c].size(); ++i) {
						std::vector<double> hog_features;
						computeHOG(trainImages[c][i], 8, 2, 9, hog_features);
						if (hog_features.empty()) {
							result_file << "Error: HOG feature vector is empty!" << std::endl;
							continue;
						}
						Mat hog_mat(hog_features);
						hog_mat = hog_mat.reshape(1, 1);
						hog_mat.convertTo(hog_mat, CV_64FC1);
						try {
							hog_mat.copyTo(X.row(trainIndex));
						}
						catch (const cv::Exception& e) {
							result_file << "OpenCV exception: " << e.what() << std::endl;
						}
						y.at<double>(trainIndex, 0) = c;
						trainIndex++;
					}
				}
				result_file << "Train Index: " << trainIndex << std::endl;
				int testIndex = 0;
				for (int c = 0; c < C; ++c) {
					for (size_t i = 0; i < testImages[c].size(); ++i) {
						std::vector<double> hog_features;
						computeHOG(testImages[c][i], 8, 2, 9, hog_features);
						Mat hog_mat(hog_features);
						hog_mat = hog_mat.reshape(1, 1);
						hog_mat.convertTo(hog_mat, CV_64FC1);
						hog_mat.copyTo(X_test.row(testIndex));
						y_test.at<double>(testIndex, 0) = c;
						testIndex++;
					}
				}
				result_file << "Test Index: " << testIndex << std::endl;
				Mat priors(C, 1, CV_64FC1);
				Mat likelihood(C, d, CV_64FC1, Scalar(1));

				for (int c = 0; c < C; ++c) {
					Mat classSamples = X.rowRange(trainIndex * c / C, trainIndex * (c + 1) / C);
					int classCount = classSamples.rows;

					priors.at<double>(c, 0) = static_cast<double>(classCount) / totalTrainSamples;
					for (int j = 0; j < 4320; ++j) {
						double count = 0.0;
						for (int k = 0; k < classSamples.rows; ++k) {
							if (classSamples.at<double>(k, j) != 0) {
								count++;

							}
						}
						if (count == 0) {
							likelihood.at<double>(c, j) = (count + 1) / (classCount + C);
						}
						else {
							likelihood.at<double>(c, j) = count / classCount;
						}
					}
				}

				int correct = 0, total = 0;
				Mat confusionMatrix = Mat::zeros(C, C, CV_32S);

				for (int i = 0; i < X_test.rows; ++i) {
					Mat img = X_test.row(i).reshape(1, 4320);
					int trueClass = static_cast<int>(y_test.at<double>(i, 0));
					int predictedClass = classifyBayes(img, priors, likelihood);

					confusionMatrix.at<int>(predictedClass, trueClass)++;
					if (trueClass == predictedClass) correct++;
					total++;
				}

				double errorRate = 1.0 - (double)correct / total;

				result_file << "Train size: " << totalTrainSamples << std::endl;
				result_file << "Test size: " << totalTestSamples << std::endl;
				result_file << "Error Rate: " << errorRate << std::endl;
				result_file << "Confusion Matrix: " << std::endl << confusionMatrix << std::endl;
			}
			result_file.close();
		}
		break;
		case 6:
		{
			char fname[256];
			while (openFileDlg(fname))
			{
				Mat img = imread(fname, cv::IMREAD_GRAYSCALE);
				imshow("Input", img);
				Mat binarized = basicGlobalThresholding(img);
				imshow("Binarized", binarized);
				std::ofstream result_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/evaluation_results_single_image_hog.txt", std::ios::app);
				if (result_file.is_open()) {
					result_file << "\n--------------------------------------\n";
					const int C = 36;
					const int d = 4320;
					show = true;
					Mat resized;
					cv::resize(binarized, resized, cv::Size(128, 64));
					imshow("Resized", resized);
					std::vector<double> hog_features;
					computeHOG(resized, 8, 2, 9, hog_features);
					if (hog_features.empty()) {
						result_file << "Error: HOG feature vector is empty!" << std::endl;
						break;
					}
					show = false;
					std::vector<Mat> trainImages[C];
					Mat X, y;
					srand(static_cast<unsigned int>(time(0)));

					for (int c = 0; c < C; ++c) {
						std::string folderName;
						std::string prefix;

						if (c < 10) {
							folderName = "digit_" + std::to_string(c);
							prefix = std::to_string(c);
						}
						else {
							folderName = "letter_" + std::string(1, char('A' + (c - 10)));
							prefix = std::string(1, char('A' + (c - 10)));
						}

						std::string folderPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/characters/" + folderName;
						std::vector<std::string> filenames;
						for (int index = 1; ; ++index) {
							char fname[256];
							sprintf(fname, "%s/%s_%d.png", folderPath.c_str(), prefix.c_str(), index);
							Mat img = imread(fname, 0);
							if (img.empty()) break;
							filenames.push_back(std::string(fname));
						}

						std::random_device rd;
						std::mt19937 g(rd());
						std::shuffle(filenames.begin(), filenames.end(), g);

						int numTrain = static_cast<int>(filenames.size() * 0.8);

						for (int i = 0; i < numTrain; ++i) {
							Mat img = imread(filenames[i], 0);
							Mat resized;
							cv::resize(img, resized, cv::Size(128, 64));
							Mat binarized = basicGlobalThresholding(resized);
							trainImages[c].push_back(binarized);
						}
					}

					int totalTrainSamples = 0, totalTestSamples = 0;
					for (int c = 0; c < C; ++c) {
						totalTrainSamples += trainImages[c].size();
					}
					X = Mat(totalTrainSamples, 4320, CV_64FC1);
					y = Mat(totalTrainSamples, 1, CV_64FC1);
					Mat X_test(1, d, CV_64FC1);
					Mat hog_mat(hog_features);
					hog_mat = hog_mat.reshape(1, 1);
					hog_mat.convertTo(hog_mat, CV_64FC1);
					hog_mat.copyTo(X_test.row(0));

					int trainIndex = 0;
					for (int c = 0; c < C; ++c) {
						for (size_t i = 0; i < trainImages[c].size(); ++i) {
							std::vector<double> hog_features;
							computeHOG(trainImages[c][i], 8, 2, 9, hog_features);
							if (hog_features.empty()) {
								result_file << "Error: HOG feature vector is empty!" << std::endl;
								continue;
							}
							Mat hog_mat(hog_features);
							hog_mat = hog_mat.reshape(1, 1);
							hog_mat.convertTo(hog_mat, CV_64FC1);
							try {
								hog_mat.copyTo(X.row(trainIndex));
							}
							catch (const cv::Exception& e) {
								result_file << "OpenCV exception: " << e.what() << std::endl;
							}
							y.at<double>(trainIndex, 0) = c;
							trainIndex++;
						}
					}
					Mat priors(C, 1, CV_64FC1);
					Mat likelihood(C, d, CV_64FC1);
					for (int c = 0; c < C; ++c) {
						Mat classSamples = X.rowRange(trainIndex * c / C, trainIndex * (c + 1) / C);
						int classCount = classSamples.rows;
						priors.at<double>(c, 0) = static_cast<double>(classCount) / totalTrainSamples;
						for (int j = 0; j < 4320; ++j) {
							double count = 0.0;
							for (int k = 0; k < classSamples.rows; ++k) {
								count += (classSamples.at<double>(k, j) == 255 ? 1.0 : 0.0);
							}
							likelihood.at<double>(c, j) = (count == 0) ? (count + 1) / (classCount + C) : count / classCount;
						}
					}
					int predictedClass = classifyBayes(X_test.row(0), priors, likelihood);
					result_file << "Predicted class: " << predictedClass << std::endl;
				}
				result_file.close();
			}
		}
		break;
		case 7:
		{
			std::ofstream result_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/evaluation_results_characters_voting.txt", std::ios::app);
			if (result_file.is_open()) {
				result_file << "\n--------------------------------------\n";
				const int C = 36;
				const int d = 128 * 64;
				show = false;
				std::vector<Mat> trainImages, testImages;
				std::vector<int> trainLabels, testLabels;
				int trainIndex = 0, testIndex = 0;
				std::map<int, std::pair<int, int>> classIndexes;

				srand(static_cast<unsigned int>(time(0)));

				for (int c = 0; c < C; ++c) {
					std::string folderName;
					std::string prefix;

					if (c < 10) {
						folderName = "digit_" + std::to_string(c);
						prefix = std::to_string(c);
					}
					else {
						folderName = "letter_" + std::string(1, char('A' + (c - 10)));
						prefix = std::string(1, char('A' + (c - 10)));
					}
					int classStartIndex = trainIndex;
					std::string folderPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/characters/" + folderName;
					std::vector<std::string> filenames;
					for (int index = 1; ; ++index) {
						char fname[256];
						sprintf(fname, "%s/%s_%d.png", folderPath.c_str(), prefix.c_str(), index);
						Mat img = imread(fname, 0);
						if (img.empty()) break;
						filenames.push_back(std::string(fname));
					}
					std::random_device rd;
					std::mt19937 g(rd());
					std::shuffle(filenames.begin(), filenames.end(), g);
					size_t trainSize = filenames.size() * 0.8;
					for (size_t i = 0; i < filenames.size(); ++i) {
						cv::Mat img = cv::imread(filenames[i], 0);
						cv::resize(img, img, cv::Size(128, 64));
						img = basicGlobalThresholding(img);
						std::vector<double> hogFeatures;
						computeHOG(img, 8, 2, 9, hogFeatures);
						cv::Mat featureMat(hogFeatures);
						featureMat = featureMat.reshape(1, 1);
						featureMat.convertTo(featureMat, CV_32F);
						if (i < trainSize) {
							trainImages.push_back(featureMat);
							trainLabels.push_back(c);
							trainIndex++;
						}
						else {
							testImages.push_back(featureMat);
							testLabels.push_back(c);
							testIndex++;
						}
					}
					int classEndIndex = trainIndex - 1;
					classIndexes[c] = { classStartIndex, classEndIndex };
				}

				cv::Mat X_train, y_train, X_test, y_test;
				cv::vconcat(trainImages, X_train);
				cv::Mat(trainLabels).convertTo(y_train, CV_32S);
				cv::vconcat(testImages, X_test);
				cv::Mat(testLabels).convertTo(y_test, CV_32S);
				cv::Mat priors(C, 1, CV_64FC1);
				cv::Mat likelihood(C, d, CV_64FC1, cv::Scalar(1.0));

				for (int c = 0; c < C; ++c) {
					int classStartIndex = classIndexes[c].first;
					int classEndIndex = classIndexes[c].second;
					Mat classSamples = X_train.rowRange(classStartIndex, classEndIndex + 1);
					int classCount = classSamples.rows;
					result_file << "Class " << c << " count = " << classCount << std::endl;
					classSamples.convertTo(classSamples, CV_64F);
					priors.at<double>(c, 0) = static_cast<double>(classCount) / X_train.rows;
					for (int j = 0; j < 4320; ++j) {
						double count = 0.0;
						for (int k = 0; k < classSamples.rows; ++k) {
							if (classSamples.at<double>(k, j) != 0) {
								count++;
							}
						}
						if (count == 0) {
							likelihood.at<double>(c, j) = (count + 1) / (classCount + C);
						}
						else {
							likelihood.at<double>(c, j) = count / classCount;
						}
						result_file << "Likelihood for class " << c << ", feature " << j << " = " << likelihood.at<double>(c, j) << std::endl;
					}
				}

				int correct = 0, total = 0;
				Mat confusionMatrix = Mat::zeros(C, C, CV_32S);

				std::vector<int> predictedClasses;

				for (int i = 0; i < X_test.rows; ++i) {
					Mat img = X_test.row(i).reshape(1, 4320);
					int predictedClass = classifyBayes(img, priors, likelihood);
					predictedClasses.push_back(predictedClass);
				}
				std::vector<std::shared_ptr<cv::ml::StatModel>> classifiers;

				auto svm = cv::ml::SVM::create();
				svm->setKernel(cv::ml::SVM::LINEAR);
				svm->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(svm);

				auto rf = cv::ml::RTrees::create();
				rf->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(rf);

				auto knn = cv::ml::KNearest::create();
				knn->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(knn);

				std::vector<cv::Mat> testSamplesVec;
				for (int i = 0; i < X_test.rows; ++i) {
					testSamplesVec.push_back(X_test.row(i));
				}

				std::vector<std::shared_ptr<CustomBayesClassifier>> customClassifiers;
				customClassifiers.push_back(std::make_shared<CustomBayesClassifier>(priors, likelihood));

				std::vector<cv::Mat> validationSamplesVec;
				std::vector<int> validationLabels;

				for (int i = 0; i < X_test.rows; ++i) {
					validationSamplesVec.push_back(X_test.row(i));
				}
				validationLabels = std::vector<int>(testLabels.begin(), testLabels.end());

				double bayesWeightScale = 1.2;
				std::vector<double> weights = computeClassifierWeights(
					classifiers, customClassifiers, validationSamplesVec, validationLabels, bayesWeightScale);

				std::vector<double> classifierWeights(weights.begin(), weights.begin() + classifiers.size());
				std::vector<double> customClassifierWeights(weights.begin() + classifiers.size(), weights.end());

				std::vector<int> predictions = weightedVotingClassifier(
					classifiers, customClassifiers, classifierWeights, customClassifierWeights, testSamplesVec);

				for (size_t i = 0; i < testLabels.size(); ++i) {
					if (predictions[i] == testLabels[i]) {
						correct++;
					}
					confusionMatrix.at<int>(predictions[i], testLabels[i])++;
					total++;
				}
				double accuracy = static_cast<double>(correct) / total;
				result_file << "Accuracy: " << accuracy << std::endl;
				for (int c = 0; c < C; ++c) {
					int TP = confusionMatrix.at<int>(c, c);
					int FN = cv::sum(confusionMatrix.row(c))[0] - TP;
					int FP = cv::sum(confusionMatrix.col(c))[0] - TP;
					int TN = cv::sum(confusionMatrix)[0] - (TP + FP + FN);

					double precision = TP / static_cast<double>(TP + FP);
					double recall = TP / static_cast<double>(TP + FN);
					double f1Score = 2 * (precision * recall) / (precision + recall);

					result_file << "Class " << c << ": Precision = " << precision
						<< ", Recall = " << recall
						<< ", F1-Score = " << f1Score << std::endl;
				}

			}
			result_file.close();
		}
		break;
		case 8:
		{
			show = false;
			std::ofstream result_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/evaluation_results_case8.txt", std::ios::app);
			if (result_file.is_open()) {
				result_file << "\n--------------------------------------\n";

				const std::string baseTrainPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/characters/";
				const std::string baseTestPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/charactersResulted/";
				//const std::string baseTestPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/datasetOne/";
				const int C = 36; // 0-9 digits + 26 letters
				const int d = 128 * 64; // feature dimensions
				std::vector<Mat> trainImages, testImages;
				std::vector<int> trainLabels, testLabels;
				int trainIndex = 0, testIndex = 0;
				std::map<int, std::pair<int, int>> classIndexes;

				result_file << "Training phase" << std::endl;
				for (int c = 0; c < C; ++c) {
					std::string folderName = (c < 10)
						? "digit_" + std::to_string(c)
						: "letter_" + std::string(1, char('A' + (c - 10)));
					std::string folderPath = baseTrainPath + folderName;
					//result_file << "Folder " << folderPath << std::endl;
					std::string prefix = (c < 10)
						? std::to_string(c)
						: std::string(1, char('A' + (c - 10)));
					int classStartIndex = trainIndex;
					for (int index = 1; ; ++index) {
						char fname[256];
						sprintf(fname, "%s/%s_%d.png", folderPath.c_str(), prefix.c_str(), index);
						Mat img = imread(fname, 0);
						if (img.empty()) break;

						cv::resize(img, img, cv::Size(128, 64));
						img = basicGlobalThresholding(img);

						std::vector<double> hogFeatures;
						computeHOG(img, 8, 2, 9, hogFeatures);

						cv::Mat featureMat(hogFeatures);
						featureMat = featureMat.reshape(1, 1);
						featureMat.convertTo(featureMat, CV_32F);
						//result_file<< "Computed HOG Features size: " << hogFeatures.size() << std::endl;

						trainImages.push_back(featureMat);
						trainLabels.push_back(c);
						trainIndex++;
					}
					int classEndIndex = trainIndex - 1;
					classIndexes[c] = { classStartIndex, classEndIndex };
				}

				cv::Mat X_train, y_train;
				cv::vconcat(trainImages, X_train);
				cv::Mat(trainLabels).convertTo(y_train, CV_32S);
				std::vector<CandidateResult> candidateResults;
				std::vector<TestImageInfo> testImageInfoVec;


				result_file << "Testing phase" << std::endl;
				for (const auto& plateFolder : std::filesystem::directory_iterator(baseTestPath)) {
					if (std::filesystem::is_directory(plateFolder)) {
						std::string plateName = plateFolder.path().filename().string(); // Get folder name (e.g., "007PLATECOM")

						// Ensure plateName is not empty
						if (plateName.empty()) {
							continue;
						}
						//result_file << "Plate: " << plateName << std::endl;
						// Process each candidate folder inside the plate folder
						for (const auto& candidateFolder : std::filesystem::directory_iterator(plateFolder.path())) {
							if (std::filesystem::is_directory(candidateFolder)) {
								//result_file << candidateFolder << std::endl;
								std::string folderName = candidateFolder.path().filename().string();
								int candidateIndex = -1; // Default to -1 for safety
								if (folderName.rfind("candidate", 0) == 0) { // Check if folder name starts with "candidate"
									try {
										candidateIndex = std::stoi(folderName.substr(9)); // Extract the numeric part
									}
									catch (const std::invalid_argument& e) {
										std::cerr << "Invalid folder name: " << folderName << std::endl;
										continue;
									}
									catch (const std::out_of_range& e) {
										std::cerr << "Out-of-range number in folder name: " << folderName << std::endl;
										continue;
									}
								}

								int charIndex = 0; // For keeping track of which character we're dealing with
								// Iterate through the images in the candidate folder
								for (const auto& imgFile : std::filesystem::directory_iterator(candidateFolder.path())) {
									if (imgFile.path().extension() == ".png") {
										Mat img = imread(imgFile.path().string(), 0);
										if (img.empty()) continue;

										// Ensure charIndex is within bounds of plateName length
										if (charIndex >= plateName.size()) {
											break; // Exit if charIndex exceeds plateName length
										}
										img = repeatClosingVertical(img, 1);
										cv::resize(img, img, cv::Size(128, 64));
										//if (img.rows < 25) {
										//	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
										//	img = basicGlobalThresholding(img); // Apply thresholding to binarize the image
										//	img = repeatClosingVertical(img, 1);
										//	cv::resize(img, img, cv::Size(128, 64)); // Resize the image to match input size
										//	img = repeatOpening(img, 1);
										//}
										//else {
										//	img = repeatClosingVertical(img, 1);
										//	cv::resize(img, img, cv::Size(128, 64)); // Resize the image to match input size
										//}

										img = basicGlobalThresholding(img); // Apply thresholding to binarize the image										
										std::vector<double> hogFeatures;
										computeHOG(img, 8, 2, 9, hogFeatures); // Extract HOG features

										cv::Mat featureMat(hogFeatures);
										featureMat = featureMat.reshape(1, 1); // Reshape into a single row
										featureMat.convertTo(featureMat, CV_32F); // Convert to 32-bit float

										// Map each character in the plate name to its corresponding label
										char character = plateName[charIndex];
										int label = -1; // Initialize label

										// Handle digits and letters
										if (character >= '0' && character <= '9') {
											label = character - '0'; // Label digits 0-9 as 0-9
										}
										else if (character >= 'A' && character <= 'Z') {
											label = character - 'A' + 10; // Label letters A-Z as 10-35
										}
										std::string imageName = imgFile.path().stem().string();
										// Ensure the label is valid
										if (label != -1) {
											testImages.push_back(featureMat);
											testLabels.push_back(label);
											TestImageInfo info = { testIndex, plateName, candidateIndex, imageName };
											testImageInfoVec.push_back(info);
										}
										charIndex++;
										testIndex++;
									}
								}
							}
						}
					}
				}

				cv::Mat X_test, y_test;
				cv::vconcat(testImages, X_test);
				cv::Mat(testLabels).convertTo(y_test, CV_32S);
				cv::Mat priors(C, 1, CV_64FC1);
				cv::Mat likelihood(C, d, CV_64FC1, cv::Scalar(1.0));

				result_file << "Bayes Classifier" << std::endl;
				//result_file << "Train Index = " << trainIndex << std::endl;
				int countZero = 0, countNonZero = 0;
				for (int c = 0; c < C; ++c) {
					int classStartIndex = classIndexes[c].first;
					int classEndIndex = classIndexes[c].second;
					Mat classSamples = X_train.rowRange(classStartIndex, classEndIndex + 1);
					int classCount = classSamples.rows;
					classSamples.convertTo(classSamples, CV_64F);

					cv::Mat mean, stddev;
					cv::meanStdDev(classSamples, mean, stddev);

					priors.at<double>(c, 0) = static_cast<double>(classCount) / X_train.rows;

					std::vector<double> feature_likelihoods(4320, 0.0);

					for (int j = 0; j < 4320; ++j) {
						double feature_value = classSamples.at<double>(0, j); // Example for the first sample
						double mean_value = mean.at<double>(j);
						double stddev_value = stddev.at<double>(j);

						// Prevent zero variance
						if (stddev_value < 1e-6) {
							stddev_value = 1e-6;  // Minimum stddev to avoid division by zero
						}

						// Ensure no NaN values in mean or stddev
						if (std::isnan(mean_value) || std::isnan(stddev_value)) {
							feature_likelihoods[j] = 1e-10;  // Default to a very small value
							continue;
						}

						// Calculate log-likelihood
						double log_likelihood = -std::log(stddev_value * sqrt(2 * CV_PI)) -
							0.5 * pow((feature_value - mean_value) / stddev_value, 2);

						// Exponential transformation to obtain likelihood
						double likelihood_value = exp(log_likelihood);

						// Handle edge cases
						if (std::isnan(likelihood_value) || likelihood_value < 1e-10) {
							likelihood_value = 1e-10;  // Default to a very small value
						}

						// Store likelihood
						feature_likelihoods[j] = likelihood_value;
					}

					// Normalize likelihoods across features
					double sum_likelihoods = std::accumulate(feature_likelihoods.begin(), feature_likelihoods.end(), 0.0);
					if (sum_likelihoods > 0) {
						for (int j = 0; j < 4320; ++j) {
							feature_likelihoods[j] /= sum_likelihoods; // Normalize to [0, 1]
							likelihood.at<double>(c, j) = feature_likelihoods[j];

							// Log the normalized likelihood for debugging
							/*result_file << "Normalized Likelihood for class " << c << ", feature " << j << " = "
								<< feature_likelihoods[j] << std::endl;*/
						}
					}
				}

				//result_file << "Count zero = " << countZero << "; Count non-zero = " << countNonZero << std::endl;
				int correct = 0, total = 0;
				Mat confusionMatrix = Mat::zeros(C, C, CV_32S);

				std::vector<std::shared_ptr<cv::ml::StatModel>> classifiers;

				result_file << "SVM Classifier" << std::endl;
				auto svm = cv::ml::SVM::create();
				svm->setKernel(cv::ml::SVM::LINEAR);
				svm->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(svm);

				result_file << "Random Forest Classifier" << std::endl;
				auto rf = cv::ml::RTrees::create();
				rf->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(rf);

				result_file << "KNN Classifier" << std::endl;
				auto knn = cv::ml::KNearest::create();
				knn->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(knn);

				std::vector<cv::Mat> testSamplesVec;
				for (int i = 0; i < X_test.rows; ++i) {
					testSamplesVec.push_back(X_test.row(i));
				}

				std::vector<std::shared_ptr<CustomBayesClassifier>> customClassifiers;
				customClassifiers.push_back(std::make_shared<CustomBayesClassifier>(priors, likelihood));

				std::vector<cv::Mat> validationSamplesVec;
				std::vector<int> validationLabels;

				for (int i = 0; i < X_test.rows; ++i) {
					validationSamplesVec.push_back(X_test.row(i));
				}
				validationLabels = std::vector<int>(testLabels.begin(), testLabels.end());

				double bayesWeightScale = 1.2;
				std::vector<double> weights = computeClassifierWeights(
					classifiers, customClassifiers, validationSamplesVec, validationLabels, bayesWeightScale);

				std::vector<double> classifierWeights(weights.begin(), weights.begin() + classifiers.size());
				std::vector<double> customClassifierWeights(weights.begin() + classifiers.size(), weights.end());

				result_file << "Weighted Voting" << std::endl;
				std::vector<int> predictions = weightedVotingClassifier(
					classifiers, customClassifiers, classifierWeights, customClassifierWeights, testSamplesVec);

				// use TestImageInfo to go to each row in X_test and to print the corresponding plateName and the predictions
				result_file << "Test Image Info Size = " << testImageInfoVec.size() << std::endl;
				result_file << "Predictions Size = " << predictions.size() << std::endl;
				result_file << "Predictions" << std::endl;
				std::unordered_map<std::string, std::unordered_map<int, std::string>> platePredictions;

				// Process predictions and organize them by plate
				for (int i = 0; i < predictions.size(); ++i) {
					const TestImageInfo& testInfo = testImageInfoVec[i];
					char predictedChar;
					if (predictions[i] <= 9) {
						predictedChar = '0' + predictions[i]; // Convert digit to character
					}
					else {
						predictedChar = 'A' + (predictions[i] - 10); // Convert class index to character
					}

					// Group predictions by plate and candidate
					platePredictions[testInfo.plateName][testInfo.candidateIndex] += predictedChar;
				}

				analyzePredictions(platePredictions, result_file);

			}
			result_file.close();
		}
		break;
		case 9:
		{
			show = false;
			std::ofstream result_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/evaluation_results_case9.txt", std::ios::app);
			if (result_file.is_open()) {
				result_file << "\n--------------------------------------\n";

				const std::string baseTrainPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/characters/";
				const std::string baseTestPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/charactersResulted/";
				//const std::string baseTestPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/datasetOne/";
				const int C = 36; // 0-9 digits + 26 letters
				const int d = 128 * 64; // feature dimensions
				std::vector<Mat> trainImages, testImages;
				std::vector<int> trainLabels, testLabels;
				int trainIndex = 0, testIndex = 0;
				std::map<int, std::pair<int, int>> classIndexes;

				result_file << "Training phase" << std::endl;
				for (int c = 0; c < C; ++c) {
					std::string folderName = (c < 10)
						? "digit_" + std::to_string(c)
						: "letter_" + std::string(1, char('A' + (c - 10)));
					std::string folderPath = baseTrainPath + folderName;
					std::string prefix = (c < 10)
						? std::to_string(c)
						: std::string(1, char('A' + (c - 10)));
					int classStartIndex = trainIndex;
					for (int index = 1; ; ++index) {
						char fname[256];
						sprintf(fname, "%s/%s_%d.png", folderPath.c_str(), prefix.c_str(), index);
						Mat img = imread(fname, 0);
						if (img.empty()) break;

						cv::resize(img, img, cv::Size(128, 64));
						img = basicGlobalThresholding(img); // Thresholding applied

						// Using pixel-based features instead of HOG
						cv::Mat featureMat;
						img = img.reshape(1, 1); // Flatten the image matrix
						img.convertTo(featureMat, CV_32F); // Convert to float type

						trainImages.push_back(featureMat);
						trainLabels.push_back(c);
						trainIndex++;
					}
					int classEndIndex = trainIndex - 1;
					classIndexes[c] = { classStartIndex, classEndIndex };
				}

				cv::Mat X_train, y_train;
				cv::vconcat(trainImages, X_train);
				cv::Mat(trainLabels).convertTo(y_train, CV_32S);
				std::vector<CandidateResult> candidateResults;
				std::vector<TestImageInfo> testImageInfoVec;


				result_file << "Testing phase" << std::endl;
				for (const auto& plateFolder : std::filesystem::directory_iterator(baseTestPath)) {
					if (std::filesystem::is_directory(plateFolder)) {
						std::string plateName = plateFolder.path().filename().string(); // Get folder name

						if (plateName.empty()) {
							continue;
						}

						for (const auto& candidateFolder : std::filesystem::directory_iterator(plateFolder.path())) {
							if (std::filesystem::is_directory(candidateFolder)) {
								std::string folderName = candidateFolder.path().filename().string();
								int candidateIndex = -1; // Default to -1 for safety
								if (folderName.rfind("candidate", 0) == 0) {
									try {
										candidateIndex = std::stoi(folderName.substr(9)); // Extract numeric part
									}
									catch (const std::invalid_argument& e) {
										std::cerr << "Invalid folder name: " << folderName << std::endl;
										continue;
									}
									catch (const std::out_of_range& e) {
										std::cerr << "Out-of-range number in folder name: " << folderName << std::endl;
										continue;
									}
								}

								int charIndex = 0; // For keeping track of characters
								for (const auto& imgFile : std::filesystem::directory_iterator(candidateFolder.path())) {
									if (imgFile.path().extension() == ".png") {
										Mat img = imread(imgFile.path().string(), 0);
										if (img.empty()) continue;

										// Ensure charIndex is within bounds of plateName length
										if (charIndex >= plateName.size()) {
											break;
										}

										if (img.rows < 30) {
											cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
											img = basicGlobalThresholding(img); // Thresholding applied
											img = repeatClosingVertical(img, 3);
											cv::resize(img, img, cv::Size(128, 64)); // Resize to match input size
											img = repeatOpening(img, 1);
										}
										else {
											cv::resize(img, img, cv::Size(128, 64));
										}

										std::vector<double> pixelFeatures(img.begin<uchar>(), img.end<uchar>()); // Flatten pixels

										cv::Mat featureMat(pixelFeatures);
										featureMat = featureMat.reshape(1, 1); // Flatten it
										featureMat.convertTo(featureMat, CV_32F);

										// Map each character in the plate name to its corresponding label
										char character = plateName[charIndex];
										int label = -1;

										if (character >= '0' && character <= '9') {
											label = character - '0'; // Label digits 0-9 as 0-9
										}
										else if (character >= 'A' && character <= 'Z') {
											label = character - 'A' + 10; // Label letters A-Z as 10-35
										}

										if (label != -1) {
											testImages.push_back(featureMat);
											testLabels.push_back(label);
											TestImageInfo info = { testIndex, plateName, candidateIndex, imgFile.path().stem().string() };
											testImageInfoVec.push_back(info);
										}
										charIndex++;
										testIndex++;
									}
								}
							}
						}
					}
				}

				cv::Mat X_test, y_test;
				cv::vconcat(testImages, X_test);
				cv::Mat(testLabels).convertTo(y_test, CV_32S);
				cv::Mat priors(C, 1, CV_64FC1);
				cv::Mat likelihood(C, d, CV_64FC1, cv::Scalar(1.0));

				result_file << "Bayes Classifier" << std::endl;
				//result_file << "Train Index = " << trainIndex << std::endl;
				int countZero = 0, countNonZero = 0;
				for (int c = 0; c < C; ++c) {
					int classStartIndex = classIndexes[c].first;
					int classEndIndex = classIndexes[c].second;
					Mat classSamples = X_train.rowRange(classStartIndex, classEndIndex + 1);
					int classCount = classSamples.rows;
					classSamples.convertTo(classSamples, CV_64F);

					cv::Mat mean, stddev;
					cv::meanStdDev(classSamples, mean, stddev);

					priors.at<double>(c, 0) = static_cast<double>(classCount) / X_train.rows;

					std::vector<double> feature_likelihoods(d, 0.0);

					for (int j = 0; j < d; ++j) {
						double feature_value = classSamples.at<double>(0, j); // Example for the first sample
						double mean_value = mean.at<double>(j);
						double stddev_value = stddev.at<double>(j);

						// Prevent zero variance
						if (stddev_value < 1e-6) {
							stddev_value = 1e-6;  // Minimum stddev to avoid division by zero
						}

						// Ensure no NaN values in mean or stddev
						if (std::isnan(mean_value) || std::isnan(stddev_value)) {
							feature_likelihoods[j] = 1e-10;  // Default to a very small value
							continue;
						}

						// Calculate log-likelihood
						double log_likelihood = -std::log(stddev_value * sqrt(2 * CV_PI)) -
							0.5 * pow((feature_value - mean_value) / stddev_value, 2);

						// Exponential transformation to obtain likelihood
						double likelihood_value = exp(log_likelihood);

						// Handle edge cases
						if (std::isnan(likelihood_value) || likelihood_value < 1e-10) {
							likelihood_value = 1e-10;  // Default to a very small value
						}

						// Store likelihood
						feature_likelihoods[j] = likelihood_value;
					}

					// Normalize likelihoods across features
					double sum_likelihoods = std::accumulate(feature_likelihoods.begin(), feature_likelihoods.end(), 0.0);
					if (sum_likelihoods > 0) {
						for (int j = 0; j < d; ++j) {
							feature_likelihoods[j] /= sum_likelihoods; // Normalize to [0, 1]
							likelihood.at<double>(c, j) = feature_likelihoods[j];

							// Log the normalized likelihood for debugging
							/*result_file << "Normalized Likelihood for class " << c << ", feature " << j << " = "
								<< feature_likelihoods[j] << std::endl;*/
						}
					}
				}

				int correct = 0, total = 0;
				Mat confusionMatrix = Mat::zeros(C, C, CV_32S);

				std::vector<std::shared_ptr<cv::ml::StatModel>> classifiers;

				result_file << "SVM Classifier" << std::endl;
				auto svm = cv::ml::SVM::create();
				svm->setKernel(cv::ml::SVM::LINEAR);
				svm->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(svm);

				result_file << "Random Forest Classifier" << std::endl;
				auto rf = cv::ml::RTrees::create();
				rf->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(rf);

				result_file << "KNN Classifier" << std::endl;
				auto knn = cv::ml::KNearest::create();
				knn->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(knn);

				std::vector<cv::Mat> testSamplesVec;
				for (int i = 0; i < X_test.rows; ++i) {
					testSamplesVec.push_back(X_test.row(i));
				}

				std::vector<std::shared_ptr<CustomBayesClassifier>> customClassifiers;
				customClassifiers.push_back(std::make_shared<CustomBayesClassifier>(priors, likelihood));

				std::vector<cv::Mat> validationSamplesVec;
				std::vector<int> validationLabels;

				for (int i = 0; i < X_test.rows; ++i) {
					validationSamplesVec.push_back(X_test.row(i));
				}
				validationLabels = std::vector<int>(testLabels.begin(), testLabels.end());

				double bayesWeightScale = 1.2;
				std::vector<double> weights = computeClassifierWeights(
					classifiers, customClassifiers, validationSamplesVec, validationLabels, bayesWeightScale);

				std::vector<double> classifierWeights(weights.begin(), weights.begin() + classifiers.size());
				std::vector<double> customClassifierWeights(weights.begin() + classifiers.size(), weights.end());

				result_file << "Weighted Voting" << std::endl;
				std::vector<int> predictions = weightedVotingClassifier(
					classifiers, customClassifiers, classifierWeights, customClassifierWeights, testSamplesVec);

				// use TestImageInfo to go to each row in X_test and to print the corresponding plateName and the predictions
				result_file << "Test Image Info Size = " << testImageInfoVec.size() << std::endl;
				result_file << "Predictions Size = " << predictions.size() << std::endl;
				result_file << "Predictions" << std::endl;
				std::unordered_map<std::string, std::unordered_map<int, std::string>> platePredictions;

				// Process predictions and organize them by plate
				for (int i = 0; i < predictions.size(); ++i) {
					const TestImageInfo& testInfo = testImageInfoVec[i];
					char predictedChar;
					if (predictions[i] <= 9) {
						predictedChar = '0' + predictions[i]; // Convert digit to character
					}
					else {
						predictedChar = 'A' + (predictions[i] - 10); // Convert class index to character
					}

					// Group predictions by plate and candidate
					platePredictions[testInfo.plateName][testInfo.candidateIndex] += predictedChar;
				}

				analyzePredictions(platePredictions, result_file);

			}
			result_file.close();
		}
		break;
		case 10:
		{
			show = false;
			std::ofstream result_file("C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/evaluation_results_case10.txt", std::ios::app);
			if (result_file.is_open()) {
				result_file << "\n--------------------------------------\n";

				const std::string baseTrainPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/characters/";
				const std::string baseTestPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 4 Semester 1/PRS/Lab/Project/datasetOne/";
				const int C = 36; // 0-9 digits + 26 letters
				const int d = 128 * 64; // feature dimensions
				std::vector<Mat> trainImages, testImages;
				std::vector<int> trainLabels, testLabels;
				int trainIndex = 0, testIndex = 0;
				std::map<int, std::pair<int, int>> classIndexes;

				result_file << "Training phase" << std::endl;
				for (int c = 0; c < C; ++c) {
					std::string folderName = (c < 10)
						? "digit_" + std::to_string(c)
						: "letter_" + std::string(1, char('A' + (c - 10)));
					std::string folderPath = baseTrainPath + folderName;
					std::string prefix = (c < 10)
						? std::to_string(c)
						: std::string(1, char('A' + (c - 10)));
					int classStartIndex = trainIndex;
					for (int index = 1; ; ++index) {
						char fname[256];
						sprintf(fname, "%s/%s_%d.png", folderPath.c_str(), prefix.c_str(), index);
						Mat img = imread(fname, 0);
						if (img.empty()) break;

						cv::resize(img, img, cv::Size(128, 64));
						img = basicGlobalThresholding(img); // Thresholding applied

						// Using pixel-based features instead of HOG
						cv::Mat featureMat;
						img = img.reshape(1, 1); // Flatten the image matrix
						img.convertTo(featureMat, CV_32F); // Convert to float type

						trainImages.push_back(featureMat);
						trainLabels.push_back(c);
						trainIndex++;
					}
					int classEndIndex = trainIndex - 1;
					classIndexes[c] = { classStartIndex, classEndIndex };
				}

				cv::Mat X_train, y_train;
				cv::vconcat(trainImages, X_train);
				cv::Mat(trainLabels).convertTo(y_train, CV_32S);
				std::vector<CandidateResult> candidateResults;
				std::vector<TestImageInfo> testImageInfoVec;


				result_file << "Testing phase" << std::endl;
				for (const auto& plateFolder : std::filesystem::directory_iterator(baseTestPath)) {
					if (std::filesystem::is_directory(plateFolder)) {
						std::string plateName = plateFolder.path().filename().string(); // Get folder name

						if (plateName.empty()) {
							continue;
						}

						for (const auto& candidateFolder : std::filesystem::directory_iterator(plateFolder.path())) {
							if (std::filesystem::is_directory(candidateFolder)) {
								std::string folderName = candidateFolder.path().filename().string();
								int candidateIndex = -1; // Default to -1 for safety
								if (folderName.rfind("candidate", 0) == 0) {
									try {
										candidateIndex = std::stoi(folderName.substr(9)); // Extract numeric part
									}
									catch (const std::invalid_argument& e) {
										std::cerr << "Invalid folder name: " << folderName << std::endl;
										continue;
									}
									catch (const std::out_of_range& e) {
										std::cerr << "Out-of-range number in folder name: " << folderName << std::endl;
										continue;
									}
								}

								int charIndex = 0; // For keeping track of characters
								for (const auto& imgFile : std::filesystem::directory_iterator(candidateFolder.path())) {
									if (imgFile.path().extension() == ".png") {
										Mat img = imread(imgFile.path().string(), 0);
										if (img.empty()) continue;

										// Ensure charIndex is within bounds of plateName length
										if (charIndex >= plateName.size()) {
											break;
										}

										if (img.rows < 30) {
											cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
											img = basicGlobalThresholding(img); // Thresholding applied
											img = repeatClosingVertical(img, 3);
											cv::resize(img, img, cv::Size(128, 64)); // Resize to match input size
											img = repeatOpening(img, 1);
										}
										else {
											cv::resize(img, img, cv::Size(128, 64));
										}

										std::vector<double> pixelFeatures(img.begin<uchar>(), img.end<uchar>()); // Flatten pixels

										cv::Mat featureMat(pixelFeatures);
										featureMat = featureMat.reshape(1, 1); // Flatten it
										featureMat.convertTo(featureMat, CV_32F);

										// Map each character in the plate name to its corresponding label
										char character = plateName[charIndex];
										int label = -1;

										if (character >= '0' && character <= '9') {
											label = character - '0'; // Label digits 0-9 as 0-9
										}
										else if (character >= 'A' && character <= 'Z') {
											label = character - 'A' + 10; // Label letters A-Z as 10-35
										}

										if (label != -1) {
											testImages.push_back(featureMat);
											testLabels.push_back(label);
											TestImageInfo info = { testIndex, plateName, candidateIndex, imgFile.path().stem().string() };
											testImageInfoVec.push_back(info);
										}
										charIndex++;
										testIndex++;
									}
								}
							}
						}
					}
				}

				cv::Mat X_test, y_test;
				cv::vconcat(testImages, X_test);
				cv::Mat(testLabels).convertTo(y_test, CV_32S);
				cv::Mat priors(C, 1, CV_64FC1);
				cv::Mat likelihood(C, d, CV_64FC1, cv::Scalar(1.0));

				result_file << "Bayes Classifier" << std::endl;
				//result_file << "Train Index = " << trainIndex << std::endl;
				int countZero = 0, countNonZero = 0;
				for (int c = 0; c < C; ++c) {
					int classStartIndex = classIndexes[c].first;
					int classEndIndex = classIndexes[c].second;
					Mat classSamples = X_train.rowRange(classStartIndex, classEndIndex + 1);
					int classCount = classSamples.rows;
					classSamples.convertTo(classSamples, CV_64F);

					cv::Mat mean, stddev;
					cv::meanStdDev(classSamples, mean, stddev);

					priors.at<double>(c, 0) = static_cast<double>(classCount) / X_train.rows;

					std::vector<double> feature_likelihoods(d, 0.0);

					for (int j = 0; j < d; ++j) {
						double feature_value = classSamples.at<double>(0, j); // Example for the first sample
						double mean_value = mean.at<double>(j);
						double stddev_value = stddev.at<double>(j);

						// Prevent zero variance
						if (stddev_value < 1e-6) {
							stddev_value = 1e-6;  // Minimum stddev to avoid division by zero
						}

						// Ensure no NaN values in mean or stddev
						if (std::isnan(mean_value) || std::isnan(stddev_value)) {
							feature_likelihoods[j] = 1e-10;  // Default to a very small value
							continue;
						}

						// Calculate log-likelihood
						double log_likelihood = -std::log(stddev_value * sqrt(2 * CV_PI)) -
							0.5 * pow((feature_value - mean_value) / stddev_value, 2);

						// Exponential transformation to obtain likelihood
						double likelihood_value = exp(log_likelihood);

						// Handle edge cases
						if (std::isnan(likelihood_value) || likelihood_value < 1e-10) {
							likelihood_value = 1e-10;  // Default to a very small value
						}

						// Store likelihood
						feature_likelihoods[j] = likelihood_value;
					}

					// Normalize likelihoods across features
					double sum_likelihoods = std::accumulate(feature_likelihoods.begin(), feature_likelihoods.end(), 0.0);
					if (sum_likelihoods > 0) {
						for (int j = 0; j < d; ++j) {
							feature_likelihoods[j] /= sum_likelihoods; // Normalize to [0, 1]
							likelihood.at<double>(c, j) = feature_likelihoods[j];

							// Log the normalized likelihood for debugging
							/*result_file << "Normalized Likelihood for class " << c << ", feature " << j << " = "
								<< feature_likelihoods[j] << std::endl;*/
						}
					}
				}
				//result_file << "Count zero = " << countZero << "; Count non-zero = " << countNonZero << std::endl;
				int correct = 0, total = 0;
				Mat confusionMatrix = Mat::zeros(C, C, CV_32S);

				std::vector<std::shared_ptr<cv::ml::StatModel>> classifiers;

				result_file << "SVM Classifier" << std::endl;
				auto svm = cv::ml::SVM::create();
				svm->setKernel(cv::ml::SVM::LINEAR);
				svm->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(svm);

				result_file << "Random Forest Classifier" << std::endl;
				auto rf = cv::ml::RTrees::create();
				rf->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(rf);

				result_file << "KNN Classifier" << std::endl;
				auto knn = cv::ml::KNearest::create();
				knn->train(X_train, cv::ml::ROW_SAMPLE, y_train);
				classifiers.push_back(knn);

				std::vector<cv::Mat> testSamplesVec;
				for (int i = 0; i < X_test.rows; ++i) {
					testSamplesVec.push_back(X_test.row(i));
				}

				std::vector<std::shared_ptr<CustomBayesClassifier>> customClassifiers;
				customClassifiers.push_back(std::make_shared<CustomBayesClassifier>(priors, likelihood));

				std::vector<cv::Mat> validationSamplesVec;
				std::vector<int> validationLabels;

				for (int i = 0; i < X_test.rows; ++i) {
					validationSamplesVec.push_back(X_test.row(i));
				}
				validationLabels = std::vector<int>(testLabels.begin(), testLabels.end());

				double bayesWeightScale = 1.2;
				std::vector<double> weights = computeClassifierWeights(
					classifiers, customClassifiers, validationSamplesVec, validationLabels, bayesWeightScale);

				std::vector<double> classifierWeights(weights.begin(), weights.begin() + classifiers.size());
				std::vector<double> customClassifierWeights(weights.begin() + classifiers.size(), weights.end());

				result_file << "Weighted Voting" << std::endl;
				std::vector<int> predictions = weightedVotingClassifier(
					classifiers, customClassifiers, classifierWeights, customClassifierWeights, testSamplesVec);

				// use TestImageInfo to go to each row in X_test and to print the corresponding plateName and the predictions
				result_file << "Test Image Info Size = " << testImageInfoVec.size() << std::endl;
				result_file << "Predictions Size = " << predictions.size() << std::endl;
				result_file << "Predictions" << std::endl;
				std::unordered_map<std::string, std::unordered_map<int, std::string>> platePredictions;

				// Process predictions and organize them by plate
				for (int i = 0; i < predictions.size(); ++i) {
					const TestImageInfo& testInfo = testImageInfoVec[i];
					char predictedChar;
					if (predictions[i] <= 9) {
						predictedChar = '0' + predictions[i]; // Convert digit to character
					}
					else {
						predictedChar = 'A' + (predictions[i] - 10); // Convert class index to character
					}

					// Group predictions by plate and candidate
					platePredictions[testInfo.plateName][testInfo.candidateIndex] += predictedChar;
				}

				analyzePredictions(platePredictions, result_file);

			}
			result_file.close();
		}
		break;

		}
	} while (op != 0);
	return 0;
}