#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat to442_grayscale(Mat mat_image){

	uint8_t test = 0;

	Mat modified(mat_image.rows, mat_image.cols, CV_8UC1);

	for (int row = 0; row < mat_image.rows; row++){

		for (int col = 0; col < mat_image.cols; col++){
			modified.at<uint8_t>(row, col) = (
				mat_image.at<Vec3b>(row, col)[0] * 0.0722 +	//Blue
				mat_image.at<Vec3b>(row, col)[1] * 0.7152 +	//Green
				mat_image.at<Vec3b>(row, col)[2] * 0.2126);	//Red
		}
	}

	return modified;
}

int main(int argc, char** argv){

    VideoCapture cap(argv[1]);
    Mat img;
    Mat img_gray;
    Mat img_sob(1080, 1920, CV_8UC1, Scalar(0));

    int rows;
    int cols;
    int p1,p2,p3,p4,p5,p6,p7,p8,p9,G;

    while(true){
        cap.read(img);
        if (img.empty()){
            break;
	    }
        rows = img.rows;
        cols = img.cols;

        /********** begin grayscale code **********/
        img_gray = to442_grayscale(img);
        /********** end grayscale code **********/

        /********** begin sobel code **********/
        // (0,0) is top left, increasing down and to the right
        for(int i=1; i<(rows-1); i++){
            for(int j=1; j<(cols-1); j++){
                p1 = img_gray.at<uint8_t>(i-1,j-1);
                p2 = img_gray.at<uint8_t>(i-1,j);
                p3 = img_gray.at<uint8_t>(i-1,j+1);
                p4 = img_gray.at<uint8_t>(i,j-1);
                p5 = img_gray.at<uint8_t>(i,j);
                p6 = img_gray.at<uint8_t>(i,j+1);
                p7 = img_gray.at<uint8_t>(i+1,j-1);
                p8 = img_gray.at<uint8_t>(i+1,j);
                p9 = img_gray.at<uint8_t>(i+1,j+1);
                G = (abs(-p1-2*p4-p7+p3+2*p6+p9) + abs(p1+2*p2+p3-p7-2*p8-p9)) >> 3;
                img_sob.at<uint8_t>(i,j) = G;
            }
        }
        /********** end sobel code **********/

        namedWindow("Sobel", 0);
        namedWindow("Color", 0);

        resizeWindow("Sobel",cols,rows);    // 500,300
        resizeWindow("Color",500,300);   // 500,300

        moveWindow("Sobel", 0,0);  // 1000,0
        moveWindow("Color", 0,0);    // 500,0

        imshow("Sobel", img_sob);
        imshow("Color", img);

        char c=(char)waitKey(1);
        if(c==27){
            break;
        }
    }

    //cout << rows << "\n";
    //cout << cols << "\n";

	return 0;
}
