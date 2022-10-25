#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat to442_grayscale(Mat mat_image){

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

Mat to442_sobel(Mat img_g){

    Mat img_s(img_g.rows, img_g.cols, CV_8UC1);
    int p1,p2,p3,p4,p5,p6,p7,p8,p9,G;

    // (0,0) is top left, increasing down and to the right
    for(int i=1; i<(img_g.rows-1); i++){
        for(int j=1; j<(img_g.cols-1); j++){
            if(j==1){
                p1 = img_g.at<uint8_t>(i-1,j-1);
                p2 = img_g.at<uint8_t>(i-1,j);
                p3 = img_g.at<uint8_t>(i-1,j+1);
                p4 = img_g.at<uint8_t>(i,j-1);
                p5 = img_g.at<uint8_t>(i,j);
                p6 = img_g.at<uint8_t>(i,j+1);
                p7 = img_g.at<uint8_t>(i+1,j-1);
                p8 = img_g.at<uint8_t>(i+1,j);
                p9 = img_g.at<uint8_t>(i+1,j+1);
            }
            else{
                p1 = p2;
                p2 = p3;
                p3 = img_g.at<uint8_t>(i-1,j+1);
                p4 = p5;
                p5 = p6;
                p6 = img_g.at<uint8_t>(i,j+1);
                p7 = p8;
                p8 = p9;
                p9 = img_g.at<uint8_t>(i+1,j+1);
            }
            G = (abs(-p1-2*p4-p7+p3+2*p6+p9) + abs(p1+2*p2+p3-p7-2*p8-p9)) >> 3;
            img_s.at<uint8_t>(i,j) = G;
        }
    }

    return img_s;
}

int main(int argc, char** argv){

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    VideoCapture cap(argv[1]);
    Mat img;
    Mat img_gray;
    Mat img_sobel;

    while(true){

        // Get video frame, break while loop if no more frames
        cap.read(img);
        if (img.empty()){
            break;
	    }

        // Run custom grayscale and sobel functions
        img_gray = to442_grayscale(img);
        img_sobel = to442_sobel(img_gray);

        // Display image windows
        namedWindow("Sobel", 0);
        namedWindow("Color", 0);
        resizeWindow("Sobel",img_sobel.cols,img_sobel.rows);    // 500,300
        resizeWindow("Color",500,300);   // 500,300
        moveWindow("Sobel", 0,0);  // 1000,0
        moveWindow("Color", 0,0);    // 500,0
        imshow("Sobel", img_sobel);
        imshow("Color", img);

        // Stop program is "ESC" is pressed
        char c=(char)waitKey(1);
        if(c==27){
            break;
        }
    }

    // Display timing results
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << "\n";

	return 0;
}
