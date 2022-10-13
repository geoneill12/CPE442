#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

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
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
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