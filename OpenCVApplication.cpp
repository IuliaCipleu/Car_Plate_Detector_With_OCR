// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <random>
#include <baseapi.h>
#include <allheaders.h>
#pragma comment(lib, "comdlg32.lib")  // for GetOpenFileNameA
#pragma comment(lib, "shell32.lib")   // for SHGetPathFromIDListA and SHBrowseForFolderA
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <regex>
#include <iostream>
#include <fstream>
#include <filesystem> // For directory operations
#include <opencv2/opencv.hpp>

using namespace cv;
//namespace fs = std::filesystem;


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
	//printf("%d\n", m);
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
	Mat Gy = Mat(1, w, CV_32FC1);;
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

Pix* mat8ToPix(cv::Mat* mat8)
{
	Pix* pixd = pixCreate(mat8->size().width, mat8->size().height, 8);
	for (int y = 0; y < mat8->rows; y++)
	{
		for (int x = 0; x < mat8->cols; x++)
		{
			pixSetPixel(pixd, x, y, (l_uint32)mat8->at<uchar>(y, x));
		}
	}
	return pixd;
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
		int g2 = sum2 / N2;
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

std::vector<Mat> segmentCharacters(const Mat& plateImage) {
	std::vector<Mat> characters;

	Mat grayPlate;
	RGBToGrayscale(plateImage, grayPlate);

	int blockSize = 11;
	double c = 2.0;
	Mat thresholded = adaptiveThresholding(grayPlate, blockSize, c);
	Mat thresholded2;
	adaptiveThreshold(grayPlate, thresholded2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);
	if (show) {
		imshow("Adaptive Threshold", thresholded);
	}
	Mat negativeThresholded = negativeTransform(thresholded);
	if (show) {
		imshow("Negative Adaptive Threshold", negativeThresholded);
	}
	Mat dilatedNegativeThresholded = repeatDilationHorizontal(negativeThresholded, 0);
	if (show) {
		imshow("Dilated Negative Adaptive Threshold", dilatedNegativeThresholded);
	}
	std::vector<Vec4i> hierarchy;
	std::vector<std::vector<Point>> contours;
	findContours(dilatedNegativeThresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours) {
		if (!contour.empty()) { 
			double area = contourArea(contour);
			if (area > 0) { 
				if (area > 20) { 
					Rect myBoundingRect = boundingRect(contour);
					myBoundingRect &= Rect(0, 0, grayPlate.cols, grayPlate.rows);
					if (myBoundingRect.width > 0 && myBoundingRect.height > 0) { 
						double aspectRatio = static_cast<double>(myBoundingRect.width) / myBoundingRect.height;
						if (aspectRatio > 0.2 && aspectRatio < 1.3) {
							Mat character = grayPlate(myBoundingRect);
							Mat negativeCharacter = negativeTransform(character);
							if (show) {
								imshow("Negative character found", negativeCharacter);
							}
							characters.push_back(negativeCharacter);
						}
					}
				}
			}
		}
	}
	return characters;
}

//std::vector<Mat> segmentCharacters(const Mat& plateImage) {
//    std::vector<Mat> characters;
//    
//    // Pre-process the image: Convert to grayscale and apply CLAHE for contrast enhancement
//    Mat grayPlate, claheOutput;
//    RGBToGrayscale(plateImage, grayPlate);
//    Ptr<CLAHE> clahe = createCLAHE();
//    clahe->apply(grayPlate, claheOutput);
//
//    // Apply adaptive thresholding with fine-tuned parameters
//    int blockSize = 15;
//    double c = 3.0;
//    Mat thresholded = adaptiveThresholding(claheOutput, blockSize, c);
//
//    // Apply morphological operations
//    Mat morphed;
//    morphologyEx(thresholded, morphed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
//
//    // Find contours
//    std::vector<std::vector<Point>> contours;
//    std::vector<Vec4i> hierarchy;
//    findContours(morphed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    for (const auto& contour : contours) {
//        double area = contourArea(contour);
//        if (area > 50) { // Adjust area threshold
//            Rect boundingRectangle = boundingRect(contour);
//            double aspectRatio = static_cast<double>(boundingRectangle.width) / boundingRectangle.height;
//            if (aspectRatio > 0.2 && aspectRatio < 1.0) { // Narrow down aspect ratio
//                Mat character = grayPlate(boundingRectangle);
//				Mat negativeCharacter = negativeTransform(character);
//                characters.push_back(negativeCharacter);
//				if (show) {
//					imshow("Negative character found", negativeCharacter);
//				}
//            }
//        }
//    }
//    return characters;
//}

//std::vector<Mat> segmentCharacters(const Mat& plateImage) {
//	std::vector<Mat> characters;
//
//	Mat grayPlate;
//	RGBToGrayscale(plateImage, grayPlate);
//
//	// Adaptive thresholding with fine-tuned parameters
//	Mat thresholded;
//	adaptiveThreshold(grayPlate, thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);
//	imshow("Thresholded Plate", thresholded);
//
//	// Optional morphological operations
//	Mat dilated;
//	morphologyEx(thresholded, dilated, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(2, 2)));
//
//	// Use Connected Components Analysis to find character contours
//	std::vector<std::vector<Point>> contours;
//	findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//	for (const auto& contour : contours) {
//		Rect boundingBox = boundingRect(contour);
//
//		// Filter based on character size and aspect ratio
//		double aspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;
//		if (aspectRatio > 0.2 && aspectRatio < 1.2 && boundingBox.height > 15) {
//			Mat character = thresholded(boundingBox);
//			characters.push_back(character);
//			imshow("Character Segment", character);
//		}
//	}
//
//	return characters;
//}

void segmentTextRegions(const Mat& img, const std::string& outputFileName) {
	// Convert the image to grayscale
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// Perform OTSU thresholding
	Mat thresh;
	threshold(gray, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	// Define the kernel for dilation
	Mat rectKernel = getStructuringElement(MORPH_RECT, Size(18, 18));

	// Apply dilation to the thresholded image
	Mat dilation;
	dilate(thresh, dilation, rectKernel);

	// Find contours in the dilated image
	std::vector<std::vector<Point>> contours;
	findContours(dilation, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	// Create a copy of the original image for displaying the detected rectangles
	Mat imgWithRectangles = img.clone();

	// Open the output file to save extracted text regions
	std::ofstream file(outputFileName);
	if (!file.is_open()) {
		std::cerr << "Failed to open the output file." << std::endl;
		return;
	}
	else {
		std::cout << "File opened successfully." << std::endl;
	}

	// Write initial message to the file
	file << "Detected regions:" << std::endl;

	// Loop through each contour to draw rectangles and save the cropped regions
	for (const auto& cnt : contours) {
		// Get the bounding box for each contour
		Rect boundingBox = boundingRect(cnt);

		// Draw a rectangle on the image copy
		rectangle(imgWithRectangles, boundingBox, Scalar(0, 255, 0), 2);

		// Format the bounding box information and write to file and console
		file << "Detected region at: x=" << boundingBox.x
			<< ", y=" << boundingBox.y
			<< ", width=" << boundingBox.width
			<< ", height=" << boundingBox.height << std::endl;

		std::cout << "Detected region at: x=" << boundingBox.x
			<< ", y=" << boundingBox.y
			<< ", width=" << boundingBox.width
			<< ", height=" << boundingBox.height << std::endl;
	}

	file.close();

	// Display the result with rectangles around detected text regions
	imshow("Detected Text Regions", imgWithRectangles);
	waitKey(0);
}

Mat prepareForSegmentation(const Mat& plateImage) {
	std::vector<Mat> characters;

	Mat grayPlate;
	RGBToGrayscale(plateImage, grayPlate);

	int blockSize = 11;
	double c = 2.0;
	Mat thresholded = adaptiveThresholding(grayPlate, blockSize, c);
	Mat thresholded2;
	adaptiveThreshold(grayPlate, thresholded2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);
	if (show) {
		imshow("Adaptive Threshold", thresholded);
	}
	Mat negativeThresholded = negativeTransform(thresholded);
	if (show) {
		imshow("Negative Adaptive Threshold", negativeThresholded);
	}
	Mat dilatedNegativeThresholded = repeatDilationHorizontal(negativeThresholded, 0);
	if (show) {
		imshow("Dilated Negative Adaptive Threshold", dilatedNegativeThresholded);
	}
	return dilatedNegativeThresholded;
}

Mat findROI(const Mat& img) {
	Mat hist_vert, hist_hor;
	// Calculate the sum of intensities along each row
	Mat row_sums(img.rows, 1, CV_32SC1);
	for (int y = 0; y < img.rows; y++) {
		int sum = 0;
		for (int x = 0; x < img.cols; x++) {
			sum += img.at<uchar>(y, x);
		}
		row_sums.at<int>(y, 0) = sum;
	}

	// Calculate the sum of intensities along each column
	Mat col_sums(1, img.cols, CV_32SC1);
	for (int x = 0; x < img.cols; x++) {
		int sum = 0;
		for (int y = 0; y < img.rows; y++) {
			sum += img.at<uchar>(y, x);
		}
		col_sums.at<int>(0, x) = sum;
	}

	// Find the boundaries of the ROI
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

	// Crop the image
	Mat cropped_img = img(Rect(left, top, right - left + 1, bottom - top + 1));

	// Display the cropped image
	imshow("Cropped Image", cropped_img);
	return cropped_img;
}

Mat extractLicensePlate(const Mat& binary) {
	// Step 1: Calculate the vertical histogram (sum of non-zero pixels in each column)
	Mat vertical_hist(1, binary.cols, CV_32S, Scalar(0));
	for (int col = 0; col < binary.cols; ++col) {
		vertical_hist.at<int>(0, col) = countNonZero(binary.col(col));
	}

	// Step 2: Find left and right boundaries using a threshold
	double maxVal;
	minMaxLoc(vertical_hist, nullptr, &maxVal);
	int thresholdVertically = static_cast<int>(maxVal * 0.2); // 10% of max value to detect white area

	int left = -1, right = -1;
	for (int col = 0; col < vertical_hist.cols; ++col) {
		if (vertical_hist.at<int>(0, col) > thresholdVertically) {
			if (left == -1) {
				left = col; // Start of the white rectangle
			}
			right = col; // Continuously update the right boundary
		}
	}

	// Step 3: Calculate the horizontal histogram (sum of non-zero pixels in each row)
	Mat horizontal_hist(binary.rows, 1, CV_32S, Scalar(0));
	for (int row = 0; row < binary.rows; ++row) {
		horizontal_hist.at<int>(row, 0) = countNonZero(binary.row(row));
	}

	// Step 4: Find top and bottom boundaries using the new threshold
	int thresholdHorizontally = static_cast<int>(maxVal * 0.1);
	int top = -1, bottom = -1;
	for (int row = 0; row < horizontal_hist.rows; ++row) {
		if (horizontal_hist.at<int>(row, 0) > thresholdHorizontally) {
			if (top == -1) {
				top = row; // Start of the white rectangle
			}
			bottom = row; // Continuously update the bottom boundary
		}
	}
	Mat license_plate = binary.clone();
	// Step 5: Crop the white rectangle region
	if (left != -1 && right != -1 && top != -1 && bottom != -1) {
		Rect roi(left, top, right - left, bottom - top); // Define the region of interest
		license_plate = binary(roi); // Crop the image
		imshow("Extracted License Plate", license_plate); // Show the extracted region
	}
	else {
		printf("License plate boundaries not found!\n");
	}
	return license_plate;
}

void segmentationHorizontal(const Mat& binary) {
	Mat horizontal_hist(binary.rows, 1, CV_32S, Scalar(0));

	// Calculate the horizontal histogram (sum of non-zero pixels in each row)
	for (int row = 0; row < binary.rows; ++row) {
		horizontal_hist.at<int>(row, 0) = countNonZero(binary.row(row));
	}

	// Define a threshold to detect character regions (20% of the max value)
	double maxVal;
	minMaxLoc(horizontal_hist, nullptr, &maxVal);
	int threshold = static_cast<int>(maxVal * 0.2);

	// Detect horizontal boundaries (start and end of character regions)
	std::vector<std::pair<int, int>> character_boundaries;
	bool in_character = false;
	int start = 0;

	for (int row = 0; row < horizontal_hist.rows; ++row) {
		int pixel_sum = horizontal_hist.at<int>(row, 0);

		if (pixel_sum > threshold && !in_character) {
			in_character = true;
			start = row;
		}
		else if (pixel_sum <= threshold && in_character) {
			in_character = false;
			character_boundaries.push_back(std::make_pair(start, row));
		}
	}

	// Display each detected character as a separate image
	int char_count = 1;
	for (const auto& bounds : character_boundaries) {
		int start_row = bounds.first;
		int end_row = bounds.second;

		// Crop the character region based on row boundaries
		Mat character = binary(Rect(0, start_row, binary.cols, end_row - start_row));
		std::string window_name = "(Horizontal) Character " + std::to_string(char_count++);
		imshow(window_name, character);
	}
}

void segmentationVertical(const Mat& binary) {
	Mat vertical_hist(1, binary.cols, CV_32S, Scalar(0));
	for (int col = 0; col < binary.cols; ++col) {
		vertical_hist.at<int>(0, col) = countNonZero(binary.col(col));
	}

	// Define a threshold to detect character regions (20% of the max value)
	double maxVal;
	minMaxLoc(vertical_hist, nullptr, &maxVal);
	int threshold = static_cast<int>(maxVal * 0.2);

	// Detect character boundaries
	std::vector<std::pair<int, int>> character_boundaries;
	bool in_character = false;
	int start = 0;

	for (int col = 0; col < vertical_hist.cols; ++col) {
		int pixel_sum = vertical_hist.at<int>(0, col);

		if (pixel_sum > threshold && !in_character) {
			in_character = true;
			start = col;
		}
		else if (pixel_sum <= threshold && in_character) {
			in_character = false;
			character_boundaries.push_back(std::make_pair(start, col));
		}
	}

	// Display each character as a separate image
	int char_count = 1;
	for (const auto& bounds : character_boundaries) {
		int start_col = bounds.first;
		int end_col = bounds.second;
		printf("In for of Vertical Segmentation\n");
		Mat character = binary(Rect(start_col, 0, end_col - start_col, binary.rows));
		std::string window_name = "(Vertical) Character " + std::to_string(char_count++);
		imshow(window_name, character);
		//segmentationHorizontal(character);
	}
}


char recognizeCharacter(const Mat& character) {
	std::vector<Mat> templates;
	std::vector<char> characters;
	std::string pathToChar = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/characters/";

	// Load digit templates (0-9)
	for (int i = 0; i < 10; ++i) {
		std::string digitFolder = pathToChar + "digit_" + std::to_string(i) + "/";
		if (show) {
			std::cout << "Searching in folder: " << digitFolder << std::endl;
		}
		std::vector<cv::String> filenames;
		cv::glob(digitFolder, filenames, false);
		if (show) {
			std::cout << "Found " << filenames.size() << " files." << std::endl;
		}
		for (const auto& filename : filenames) {
			//std::cout << "Filename: " << filename << std::endl;
			Mat digitTemplate = imread(filename, IMREAD_GRAYSCALE);
			if (!digitTemplate.empty()) {
				templates.push_back(digitTemplate);
				characters.push_back('0' + i);
				//printf("%d\n", i);
			}
			else {
				printf("Error loading template: %s\n", filename.c_str());
				return ' '; // Return an empty character if template loading fails
			}
		}
	}

	// Load letter templates (A-Z)
	for (int i = 0; i < 26; ++i) {
		char letter = static_cast<char>('A' + i);
		std::string letterFolder = pathToChar + "letter_" + letter + "/";
		if (show) {
			std::cout << "Searching in folder: " << letterFolder << std::endl;
		}
		std::vector<cv::String> filenames;
		cv::glob(letterFolder, filenames, false);
		if (show) {
			std::cout << "Found " << filenames.size() << " files." << std::endl;
		}
		for (const auto& filename : filenames) {
			//std::cout << "Filename: " << filename << std::endl;
			Mat letterTemplate = imread(filename, IMREAD_GRAYSCALE);
			if (!letterTemplate.empty()) {
				templates.push_back(letterTemplate);
				characters.push_back(letter);
				//printf("%c\n", letter);
			}
			else {
				printf("Error loading template: %s\n", filename.c_str());
				return ' '; // Return an empty character if template loading fails
			}
		}
	}

	// Calculate similarity between the character and templates
	double maxSimilarity = -std::numeric_limits<double>::infinity();
	char recognizedChar = ' ';

	for (size_t i = 0; i < templates.size(); ++i) {
		double similarity = 0.0;
		try {
			Mat resizedCharacter;
			Size templateSize = templates[i].size();
			resize(character, resizedCharacter, templateSize);
			Mat result;
			matchTemplate(resizedCharacter, templates[i], result, TM_CCORR_NORMED);
			double minVal, maxVal;
			Point minLoc, maxLoc;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
			similarity = maxVal;
		}
		catch (const cv::Exception& e) {
			std::cerr << "Error comparing structural similarity with template " << i << ": " << e.what() << std::endl;
			continue;
		}

		if (similarity > maxSimilarity) {
			maxSimilarity = similarity;
			recognizedChar = characters[i];
		}
	}

	return recognizedChar;
}

std::string extractText(const Mat& plateImage) {
	std::vector<Mat> characters = segmentCharacters(plateImage);

	std::string extractedText;
	for (const auto& character : characters) {
		char recognizedChar = recognizeCharacter(character);
		extractedText += recognizedChar;
	}

	std::cout << "Extracted Text: " << extractedText << std::endl;
	return extractedText;
}

std::string getFilenameFromPath(const std::string& filepath) {
	// Find the position of the last occurrence of '/'
	size_t pos = filepath.find_last_of("/");

	// If '/' is found, return the substring after it, otherwise return the full filepath
	if (pos != std::string::npos) {
		return filepath.substr(pos + 1);
	}
	else {
		// No '/' found, return the full filepath
		return filepath;
	}
}

std::string cleanText(const std::string& text) {
	std::string cleanedText = text;

	// Remove spaces and newlines
	cleanedText.erase(std::remove_if(cleanedText.begin(), cleanedText.end(),
		[](unsigned char c) { return std::isspace(c); }),
		cleanedText.end());

	return cleanedText;
}

void writeResultsToCSV(const std::string& filename, const std::vector<std::pair<std::string, std::string>>& dataset) {
	// Create an output filestream object
	std::ofstream outputFile(filename, std::ios::app); // Append mode

	if (!outputFile.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	// Send data to the stream
	for (size_t i = 0; i < dataset.size(); ++i) {
		// Clean OCR output
		std::string cleanedOutput = cleanText(dataset[i].second);

		// Write cleaned data to CSV
		outputFile << getFilenameFromPath(dataset[i].first) << ";" << cleanedOutput << std::endl;
	}

	// Close the file
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
	// Create an output filestream object
	std::ofstream outputFile(filename, std::ios::app); // Append mode

	if (!outputFile.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	// Write data to CSV
	for (const auto& data : dataset) {
		outputFile << std::get<0>(data) << ";"
			<< std::get<1>(data) << ";"
			<< std::get<2>(data) << ";"
			<< std::get<3>(data) << ";"
			<< std::get<4>(data) << std::endl;
	}

	// Close the file
	outputFile.close();
}

void processImagesInFolder(const std::string& folderPath, const std::string& csvFilename) {
	std::vector<std::tuple<std::string, int, int, int, int>> results;
	std::vector<std::string> files = listFiles(folderPath);

	for (const auto& file : files) {
		// Display and process image
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Could not open or find the image: " << file << std::endl;
			continue;
		}
		show = true;
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
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
		GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
		imshow("Gauss", gauss);
		Canny(gauss, cannyImage, pL, pH, 3);
		imshow("Canny", cannyImage);
		Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
		imshow("My Canny", myCannyImage);
		Mat cannyNegative = negativeTransform(myCannyImage);
		imshow("Negative Canny", cannyNegative);
		Mat dilatedCanny = repeatDilationCross(cannyNegative, 3);
		imshow("Dilated Canny", dilatedCanny);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		imshow("License Plate", licensePlateImage);
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		imshow("Final Detected Plate", detectedPlate);

		waitKey();
		int userInput;
		std::cout << "Which was the result: plate detected (1), plate non-detected (2), was detected, but another area was chosen as the final one (3), or entire image given as final result(4)? ";
		std::cin >> userInput;

		int detectedFinal = (userInput == 1) ? 1 : 0;
		int nonDetected = (userInput == 2) ? 1 : 0;
		int detectedNotChosen = (userInput == 3) ? 1 : 0;
		int entireImageFinal = (userInput == 4) ? 1 : 0;

		results.emplace_back(getFilenameFromPath(file), detectedFinal, nonDetected, detectedNotChosen, entireImageFinal);

		
	}

	writeResultsToCSV5Columns(csvFilename, results);
}

void processImagesInFolder2(const std::string& folderPath, const std::string& csvFilename) {
	std::vector<std::tuple<std::string, int, int, int, int>> results;
	std::vector<std::string> files = listFiles(folderPath);

	for (const auto& file : files) {
		// Display and process image
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Could not open or find the image: " << file << std::endl;
			continue;
		}
		show = true;
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
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
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
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		imshow("License Plate", licensePlateImage);
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		imshow("Final Detected Plate", detectedPlate);

		waitKey();
		int userInput;
		std::cout << "Which was the result: plate detected (1), plate non-detected (2), was detected, but another area was chosen as the final one (3), or entire image given as final result(4)? ";
		std::cin >> userInput;

		int detectedFinal = (userInput == 1) ? 1 : 0;
		int nonDetected = (userInput == 2) ? 1 : 0;
		int detectedNotChosen = (userInput == 3) ? 1 : 0;
		int entireImageFinal = (userInput == 4) ? 1 : 0;

		results.emplace_back(getFilenameFromPath(file), detectedFinal, nonDetected, detectedNotChosen, entireImageFinal);


	}

	writeResultsToCSV5Columns(csvFilename, results);
}


bool isDirectory(const std::string& path) {
	DWORD attr = GetFileAttributesA(path.c_str());
	return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
}

int case22() {
	std::string folderPath2 = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename2 = "resultsAreaTesseractHorizontal3.csv";
	processImagesInFolder2(folderPath2, csvFilename2);
}

int case9() {
	// Get the directory path from the user
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsTesseractHorizontalDilation40.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	// Open a CSV file to store results
	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	// Write headers to the CSV file
	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	// Iterate over images in the directory
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		// Load image
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		// Preprocess image (similar to case 101)
		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
		GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
		Canny(gauss, cannyImage, pL, pH, 3);
		Mat cannyNegative = negativeTransform(cannyImage);
		Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 4);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);

		tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
		if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			exit(1);
		}

		Mat grayLicensePlate;
		RGBToGrayscale(licensePlateImage, grayLicensePlate);
		Mat thLicense = basicGlobalThresholding(grayLicensePlate);
		Mat dilatedPlate = repeatDilationHorizontal(thLicense, 0);

		Pix* pixImage = mat8ToPix(&dilatedPlate);

		api->SetImage(pixImage);

		char* outText;
		outText = api->GetUTF8Text();
		printf("OCR output:\n%s", outText);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);

		api->End();
		delete[] outText;
		pixDestroy(&pixImage);


	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case10() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsTesseractVerticalDilation34.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
		GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
		Canny(gauss, cannyImage, pL, pH, 3);
		Mat cannyNegative = negativeTransform(cannyImage);
		Mat dilatedCanny = repeatDilationVertical(cannyNegative, 3);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);

		tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
		if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			exit(1);
		}

		Mat grayLicensePlate;
		RGBToGrayscale(licensePlateImage, grayLicensePlate);
		Mat thLicense = basicGlobalThresholding(grayLicensePlate);
		Mat dilatedPlate = repeatDilationVertical(thLicense, 4);

		Pix* pixImage = mat8ToPix(&dilatedPlate);

		api->SetImage(pixImage);

		char* outText;
		outText = api->GetUTF8Text();
		printf("OCR output:\n%s", outText);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);

		api->End();
		delete[] outText;
		pixDestroy(&pixImage);


	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case11() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsTesseractCrossDilation30.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
		GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
		Canny(gauss, cannyImage, pL, pH, 3);
		Mat cannyNegative = negativeTransform(cannyImage);
		Mat dilatedCanny = repeatDilationCross(cannyNegative, 3);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);

		tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
		if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			exit(1);
		}

		Mat grayLicensePlate;
		RGBToGrayscale(licensePlateImage, grayLicensePlate);
		Mat thLicense = basicGlobalThresholding(grayLicensePlate);
		Mat dilatedPlate = repeatDilationCross(thLicense, 0);

		Pix* pixImage = mat8ToPix(&dilatedPlate);

		api->SetImage(pixImage);

		char* outText;
		outText = api->GetUTF8Text();
		printf("OCR output:\n%s", outText);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);

		api->End();
		delete[] outText;
		pixDestroy(&pixImage);


	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case12() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsCannyTesseractHorizontalDilation30.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
		Mat cannyNegative = negativeTransform(myCannyImage);
		Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 3);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);

		tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
		if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			exit(1);
		}

		Mat grayLicensePlate;
		RGBToGrayscale(licensePlateImage, grayLicensePlate);
		Mat thLicense = basicGlobalThresholding(grayLicensePlate);
		Mat dilatedPlate = repeatDilationHorizontal(thLicense, 0);

		Pix* pixImage = mat8ToPix(&dilatedPlate);

		api->SetImage(pixImage);

		char* outText;
		outText = api->GetUTF8Text();
		printf("OCR output:\n%s", outText);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);

		api->End();
		delete[] outText;
		pixDestroy(&pixImage);


	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case13() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsCannyTesseractVerticalDilation30.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
		Mat cannyNegative = negativeTransform(myCannyImage);
		Mat dilatedCanny = repeatDilationVertical(cannyNegative, 3);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);

		tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
		if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			exit(1);
		}

		Mat grayLicensePlate;
		RGBToGrayscale(licensePlateImage, grayLicensePlate);
		Mat thLicense = basicGlobalThresholding(grayLicensePlate);
		Mat dilatedPlate = repeatDilationVertical(thLicense, 0);

		Pix* pixImage = mat8ToPix(&dilatedPlate);

		api->SetImage(pixImage);

		char* outText;
		outText = api->GetUTF8Text();
		printf("OCR output:\n%s", outText);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);

		api->End();
		delete[] outText;
		pixDestroy(&pixImage);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case14() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsCannyTesseractCrossDilation30.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
		Mat cannyNegative = negativeTransform(myCannyImage);
		Mat dilatedCanny = repeatDilationCross(cannyNegative, 3);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);

		tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
		if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			exit(1);
		}

		Mat grayLicensePlate;
		RGBToGrayscale(licensePlateImage, grayLicensePlate);
		Mat thLicense = basicGlobalThresholding(grayLicensePlate);
		Mat dilatedPlate = repeatDilationCross(thLicense, 0);

		Pix* pixImage = mat8ToPix(&dilatedPlate);

		api->SetImage(pixImage);

		char* outText;
		outText = api->GetUTF8Text();
		printf("OCR output:\n%s", outText);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);

		api->End();
		delete[] outText;
		pixDestroy(&pixImage);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case15() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsOCRHorizontalDilation40.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
		GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
		Canny(gauss, cannyImage, pL, pH, 3);
		Mat cannyNegative = negativeTransform(cannyImage);
		Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 4);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		std::string outText = extractText(detectedPlate);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case16() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsOCRVerticalDilation40.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
		GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
		Canny(gauss, cannyImage, pL, pH, 3);
		Mat cannyNegative = negativeTransform(cannyImage);
		Mat dilatedCanny = repeatDilationVertical(cannyNegative, 4);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		std::string outText = extractText(detectedPlate);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case17() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsOCRCrossDilation40.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		Mat gauss;
		Mat cannyImage;
		GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
		Canny(gauss, cannyImage, pL, pH, 3);
		Mat cannyNegative = negativeTransform(cannyImage);
		Mat dilatedCanny = repeatDilationCross(cannyNegative, 4);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		std::string outText = extractText(detectedPlate);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case18() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsCannyOCRHorizontalDilation40.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
		Mat cannyNegative = negativeTransform(myCannyImage);
		Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 4);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		std::string outText = extractText(detectedPlate);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case19() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsCannyOCRVerticalDilation40.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
		Mat cannyNegative = negativeTransform(myCannyImage);
		Mat dilatedCanny = repeatDilationVertical(cannyNegative, 4);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		std::string outText = extractText(detectedPlate);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
}

int case20() {
	std::string directoryPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
	std::string csvFilename = "resultsCannyOCRCrossDilation40.csv";
	if (!isDirectory(directoryPath)) {
		std::cerr << "Directory not found: " << directoryPath << std::endl;
		return -1;
	}

	std::ofstream csvFile(csvFilename);
	if (!csvFile.is_open()) {
		std::cerr << "Failed to create CSV file: " << csvFilename << std::endl;
		return -1;
	}

	csvFile << "Image Name;OCR Output\n";
	std::vector<std::string> files = listFiles(directoryPath);
	std::vector<std::pair<std::string, std::string>> results;
	for (const std::string& file : files) {
		Mat initialImage = imread(file);
		if (initialImage.empty()) {
			std::cerr << "Failed to load image: " << file << std::endl;
			continue;
		}

		show = false;
		Mat resizedImage;
		resizeImg(initialImage, resizedImage, 750, true);
		Mat grayImage;
		RGBToGrayscale(resizedImage, grayImage);
		Mat closedImage = closingGrayscale(grayImage);
		Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
		Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
		Mat cannyNegative = negativeTransform(myCannyImage);
		Mat dilatedCanny = repeatDilationCross(cannyNegative, 4);
		Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
		Mat detectedPlate = resizedImage.clone();
		Mat plateRectMat = dilatedCanny(plateRect);
		Mat licensePlateImage = detectedPlate(plateRect);
		if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
			licensePlateImage = initialImage.clone();
		}
		rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
		std::string outText = extractText(detectedPlate);
		std::string text(outText);
		std::regex pattern("[^a-zA-Z0-9\\s-]");
		text = std::regex_replace(text, pattern, "");
		results.emplace_back(file, text);
	}
	writeResultsToCSV(csvFilename, results);
	std::cout << "Results written to " << csvFilename << std::endl;
	csvFile.close();
	return 1;
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
		printf(" 1 - Car plate recognition showing intermediate steps + my own Canny + Tesseract\n");
		printf(" 2 - Car plate recognition with result only + my own Canny + Tesseract\n");
		printf(" 3 - Car plate recognition showing intermediate steps + my own Canny + my own OCR\n");
		printf(" 4 - Car plate recognition with result + my own Canny + my own OCR\n");
		printf(" 5 - Car plate recognition showing intermediate steps + Tesseract\n");
		printf(" 6 - Car plate recognition with result only + Tesseract\n");
		printf(" 7 - Car plate recognition showing intermediate steps + my own OCR\n");
		printf(" 8 - Car plate recognition with result only + my own OCR\n");
		printf(" 9 - Create CSV for testing: resultsTesseractHorizontalXY.csv\n");
		printf(" 10 - Create CSV for testing: resultsTesseractVerticalXY.csv\n");
		printf(" 11 - Create CSV for testing: resultsTesseractCrossXY.csv\n");
		printf(" 12 - Create CSV for testing: resultsCannyTesseractHorizontalXY.csv\n");
		printf(" 13 - Create CSV for testing: resultsCannyTesseractVerticalXY.csv\n");
		printf(" 14 - Create CSV for testing: resultsCannyTesseractCrossXY.csv\n");
		printf(" 15 - Create CSV for testing: resultsOCRHorizontalXY.csv\n");
		printf(" 16 - Create CSV for testing: resultsOCRVerticalXY.csv\n");
		printf(" 17 - Create CSV for testing: resultsOCRCrossXY.csv\n");
		printf(" 18 - Create CSV for testing: resultsCannyOCRHorizontalXY.csv\n");
		printf(" 19 - Create CSV for testing: resultsCannyOCRVerticalXY.csv\n");
		printf(" 20 - Create CSV for testing: resultsCannyOCRCrossXY.csv\n");
		printf(" 21 - Create CSV for testing the area: resultsAreaCannyTesseractCross3Horizontal2.csv\n");
		printf(" 22 - Create CSV for testing the area: resultsAreaTesseractHorizontal32.csv\n");
		printf(" 23 - Car plate recognition showing intermediate steps + red detection + Tesseract\n");
		printf(" 24 - New from segmentation method, for PRS\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			char fname[MAX_PATH];
			while (openFileDlg(fname))
			{
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
				/*double k = 0.4;
				int pH = 50;
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				imshow("Gauss", gauss);
				Canny(gauss, cannyImage, pL, pH, 3);
				imshow("Canny", cannyImage);*/
				Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
				imshow("My Canny", myCannyImage);
				Mat cannyNegative = negativeTransform(myCannyImage);
				imshow("Negative Canny", cannyNegative);
				Mat dilatedCanny = repeatDilationCross(cannyNegative, 3);
				imshow("Dilated Canny", dilatedCanny);
				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);

				tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
				if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
					fprintf(stderr, "Could not initialize tesseract.\n");
					exit(1);
				}

				Mat grayLicensePlate;
				RGBToGrayscale(licensePlateImage, grayLicensePlate);
				imshow("Gray License", grayLicensePlate);
				Mat thLicense = basicGlobalThresholding(grayLicensePlate);
				imshow("Global Thresholding", thLicense);
				Mat dilatedPlate = repeatDilationHorizontal(thLicense, 2);
				imshow("Dilated Plate", dilatedPlate);
				Pix* pixImage = mat8ToPix(&dilatedPlate);

				api->SetImage(pixImage);

				char* outText;
				outText = api->GetUTF8Text();
				std::string text(outText);
				std::regex pattern("[^a-zA-Z0-9\\s-]");
				text = std::regex_replace(text, pattern, "");
				printf("OCR output:\n%s", text);

				api->End();
				delete[] outText;
				pixDestroy(&pixImage);

				waitKey();
			}
			break;
		case 2:
			while (openFileDlg(fname))
			{
				show = false;
				Mat initialImage = imread(fname);
				imshow("Input Image", initialImage);
				Mat resizedImage;
				resizeImg(initialImage, resizedImage, 750, true);
				Mat grayImage;
				RGBToGrayscale(resizedImage, grayImage);
				Mat closedImage = closingGrayscale(grayImage);
				Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
				/*double k = 0.4;
				int pH = 50;
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				Canny(gauss, cannyImage, pL, pH, 3);*/
				Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
				imshow("My Canny", myCannyImage);
				Mat cannyNegative = negativeTransform(myCannyImage);
				Mat dilatedCanny = repeatDilationCross(cannyNegative, 3);
				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);

				tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
				if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
					fprintf(stderr, "Could not initialize tesseract.\n");
					exit(1);
				}

				Mat grayLicensePlate;
				RGBToGrayscale(licensePlateImage, grayLicensePlate);
				Mat thLicense = basicGlobalThresholding(grayLicensePlate);
				Mat dilatedPlate = repeatDilationHorizontal(thLicense, 2);

				Pix* pixImage = mat8ToPix(&dilatedPlate);

				api->SetImage(pixImage);

				char* outText;
				outText = api->GetUTF8Text();
				std::string text(outText);
				std::regex pattern("[^a-zA-Z0-9\\s-]");
				text = std::regex_replace(text, pattern, "");
				printf("OCR output:\n%s", text);

				api->End();
				delete[] outText;
				pixDestroy(&pixImage);

				waitKey();
			}
			break;
		case 3:
			while (openFileDlg(fname))
			{
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
				/*double k = 0.4;
				int pH = 50;
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				imshow("Gauss", gauss);
				Canny(gauss, cannyImage, pL, pH, 3);
				imshow("Canny", cannyImage);*/
				Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
				imshow("My Canny", myCannyImage);
				Mat cannyNegative = negativeTransform(myCannyImage);
				imshow("Negative Canny", cannyNegative);
				Mat dilatedCanny = repeatDilationCross(cannyNegative, 3);
				imshow("Dilated Canny", dilatedCanny);
				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);
				extractText(licensePlateImage);
				waitKey();
			}
			break;
		case 4:
			while (openFileDlg(fname))
			{
				show = false;
				Mat initialImage = imread(fname);
				imshow("Input Image", initialImage);
				Mat resizedImage;
				resizeImg(initialImage, resizedImage, 750, true);
				Mat grayImage;
				RGBToGrayscale(resizedImage, grayImage);
				Mat closedImage = closingGrayscale(grayImage);
				Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
				/*double k = 0.4;
				int pH = 50;
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				Canny(gauss, cannyImage, pL, pH, 3);*/
				Mat myCannyImage = cannyEdgeDetection(bilateralFilteredImage, 5);
				Mat cannyNegative = negativeTransform(myCannyImage);
				Mat dilatedCanny = repeatDilationCross(cannyNegative, 3);
				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);
				extractText(detectedPlate);
				waitKey();
			}
			break;
		case 5:
			while (openFileDlg(fname))
			{
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
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				imshow("Gauss", gauss);
				Canny(gauss, cannyImage, pL, pH, 3);
				imshow("Canny", cannyImage);
				Mat cannyNegative = negativeTransform(cannyImage);
				imshow("Negative Canny", cannyNegative);
				Mat dilatedCanny = repeatDilationVertical(cannyNegative, 4);
				imshow("Dilated Canny", dilatedCanny);
				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);

				tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
				if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
					fprintf(stderr, "Could not initialize tesseract.\n");
					exit(1);
				}

				Mat grayLicensePlate;
				RGBToGrayscale(licensePlateImage, grayLicensePlate);
				imshow("Gray License", grayLicensePlate);
				Mat thLicense = basicGlobalThresholding(grayLicensePlate);
				imshow("Global Thresholding", thLicense);
				Mat dilatedPlate = repeatDilationVertical(thLicense, 1);
				imshow("Dilated Plate", dilatedPlate);
				Pix* pixImage = mat8ToPix(&dilatedPlate);

				api->SetImage(pixImage);

				char* outText;
				outText = api->GetUTF8Text();
				std::string text(outText);
				std::regex pattern("[^a-zA-Z0-9\\s-]");
				text = std::regex_replace(text, pattern, "");
				printf("OCR output:\n%s", text);

				api->End();
				delete[] outText;
				pixDestroy(&pixImage);

				waitKey();
			}
			break;
		case 6:
			while (openFileDlg(fname))
			{
				show = false;
				Mat initialImage = imread(fname);
				imshow("Input Image", initialImage);
				Mat resizedImage;
				resizeImg(initialImage, resizedImage, 750, true);
				Mat grayImage;
				RGBToGrayscale(resizedImage, grayImage);
				Mat closedImage = closingGrayscale(grayImage);
				Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
				double k = 0.4;
				int pH = 50;
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				Canny(gauss, cannyImage, pL, pH, 3);
				Mat cannyNegative = negativeTransform(cannyImage);
				Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 3);
				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);

				tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
				if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
					fprintf(stderr, "Could not initialize tesseract.\n");
					exit(1);
				}

				Mat grayLicensePlate;
				RGBToGrayscale(licensePlateImage, grayLicensePlate);
				Mat thLicense = basicGlobalThresholding(grayLicensePlate);
				Mat dilatedPlate = repeatDilationHorizontal(thLicense, 2);

				Pix* pixImage = mat8ToPix(&dilatedPlate);

				api->SetImage(pixImage);

				char* outText;
				outText = api->GetUTF8Text();
				std::string text(outText);
				std::regex pattern("[^a-zA-Z0-9\\s-]");
				text = std::regex_replace(text, pattern, "");
				printf("OCR output:\n%s", text);

				api->End();
				delete[] outText;
				pixDestroy(&pixImage);

				waitKey();
			}
			break;
		case 7:
			while (openFileDlg(fname))
			{
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
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
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
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);
				extractText(detectedPlate);
				waitKey();
			}
			break;
		case 8:
			while (openFileDlg(fname))
			{
				show = false;
				Mat initialImage = imread(fname);
				imshow("Input Image", initialImage);
				Mat resizedImage;
				resizeImg(initialImage, resizedImage, 750, true);
				Mat grayImage;
				RGBToGrayscale(resizedImage, grayImage);
				Mat closedImage = closingGrayscale(grayImage);
				Mat bilateralFilteredImage = bilateralFilterAlgorithm(closedImage, 5, 15, 15);
				double k = 0.4;
				int pH = 50;
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				Canny(gauss, cannyImage, pL, pH, 3);
				Mat cannyNegative = negativeTransform(cannyImage);
				Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 3);
				Rect plateRect = detectLicensePlate(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);
				extractText(detectedPlate);
				waitKey();
			}
			break;
		case 9:
			case9();
			break;
		case 10:
			case10();
			break;
		case 11:
			case11();
			break;
		case 12:
			case12();
			break;
		case 13:
			case13();
			break;
		case 14:
			case14();
			break;
		case 15:
			case15();
			break;
		case 16:
			case16();
			break;
		case 17:
			case17();
			break;
		case 18:
			case18();
			break;
		case 19:
			case19();
			break;
		case 20:
			case20();
			break;
		case 22:
			case22();
			break;
		case 23:
			while (openFileDlg(fname))
			{
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
				int pL = (int)k * pH;
				Mat gauss;
				Mat cannyImage;
				GaussianBlur(bilateralFilteredImage, gauss, Size(5, 5), 0.5, 0.5);
				imshow("Gauss", gauss);
				Canny(gauss, cannyImage, pL, pH, 3);
				imshow("Canny", cannyImage);
				Mat cannyNegative = negativeTransform(cannyImage);
				imshow("Negative Canny", cannyNegative);
				Mat dilatedCanny = repeatDilationHorizontal(cannyNegative, 3);
				imshow("Dilated Canny", dilatedCanny);
				Rect plateRect = detectLicensePlateWithRedComputation(dilatedCanny, resizedImage);
				Mat detectedPlate = resizedImage.clone();
				Mat plateRectMat = dilatedCanny(plateRect);
				Mat licensePlateImage = detectedPlate(plateRect);
				if (plateRect.x == 0 && plateRect.y == 0 && plateRect.width == 0 && plateRect.height == 0) {
					licensePlateImage = initialImage.clone();
				}
				imshow("License Plate", licensePlateImage);
				rectangle(detectedPlate, plateRect, Scalar(0, 255, 0), 2);
				imshow("Final Detected Plate", detectedPlate);

				tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
				if (api->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
					fprintf(stderr, "Could not initialize tesseract.\n");
					exit(1);
				}

				Mat grayLicensePlate;
				RGBToGrayscale(licensePlateImage, grayLicensePlate);
				imshow("Gray License", grayLicensePlate);
				Mat thLicense = basicGlobalThresholding(grayLicensePlate);
				imshow("Global Thresholding", thLicense);
				Mat dilatedPlate = repeatDilationHorizontal(thLicense, 2);
				imshow("Dilated Plate", dilatedPlate);
				Pix* pixImage = mat8ToPix(&dilatedPlate);

				api->SetImage(pixImage);

				char* outText;
				outText = api->GetUTF8Text();
				std::string text(outText);
				std::regex pattern("[^a-zA-Z0-9\\s-]");
				text = std::regex_replace(text, pattern, "");
				printf("OCR output:\n%s", text);

				api->End();
				delete[] outText;
				pixDestroy(&pixImage);

				waitKey();
			}
			break;
		case 21:
		{
			std::string folderPath = "C:/Users/Cipleu/Documents/IULIA/SCOALA/facultate/Year 3 Semester 2/IP/Lab/Project/dataset/imagesForTesting/plates";
			std::string csvFilename = "resultsAreaCannyTesseractCross3Horizontal2.csv";
			processImagesInFolder(folderPath, csvFilename);
		}
			break;
		case 24: 
			while (openFileDlg(fname))
		{
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
			Mat dilatedPlate = repeatDilationHorizontal(thLicense, 2);
			imshow("Dilated Plate", dilatedPlate);
			Mat roi = extractLicensePlate(dilatedPlate);
			segmentationVertical(roi);

			waitKey();
		}
			   break;
		}	
	} while (op != 0);
	return 0;
}