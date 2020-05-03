#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("D:/images/lena.png");
	imshow("input", src);
	waitKey(0);
	destroyAllWindows();
	return 0;
}