#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <pthread.h>

using namespace cv;

//pthread_barrier_t barrier;

Mat img_color;
Mat img_gray(1080, 1920, CV_8UC1);
Mat img_sobel(1080, 1920, CV_8UC1);

void grayscale_442(int row_start, int row_end, int col_start, int col_end){

	for (int row = row_start; row <= row_end; row++){
		for (int col = col_start; col <= col_end; col++){
			img_gray.at<uint8_t>(row, col) = (
				img_color.at<Vec3b>(row, col)[0] * 0.0722 +	//Blue
				img_color.at<Vec3b>(row, col)[1] * 0.7152 +	//Green
				img_color.at<Vec3b>(row, col)[2] * 0.2126);	//Red*/
		}
	}
}

void sobel_442(int row_start, int row_end, int col_start, int col_end){

    int p1,p2,p3,p4,p5,p6,p7,p8,p9,G;

    // (0,0) is top left, increasing down and to the right
    for(int i=row_start; i<=row_end; i++){
        for(int j=col_start; j<=col_end; j++){
            if(j==1){
                p1 = img_gray.at<uint8_t>(i-1,j-1);
                p2 = img_gray.at<uint8_t>(i-1,j);
                p3 = img_gray.at<uint8_t>(i-1,j+1);
                p4 = img_gray.at<uint8_t>(i,j-1);
                p5 = img_gray.at<uint8_t>(i,j);
                p6 = img_gray.at<uint8_t>(i,j+1);
                p7 = img_gray.at<uint8_t>(i+1,j-1);
                p8 = img_gray.at<uint8_t>(i+1,j);
                p9 = img_gray.at<uint8_t>(i+1,j+1);
            }
            else{
                p1 = p2;
                p2 = p3;
                p3 = img_gray.at<uint8_t>(i-1,j+1);
                p4 = p5;
                p5 = p6;
                p6 = img_gray.at<uint8_t>(i,j+1);
                p7 = p8;
                p8 = p9;
                p9 = img_gray.at<uint8_t>(i+1,j+1);
            }
            G = (abs(-p1-2*p4-p7+p3+2*p6+p9) + abs(p1+2*p2+p3-p7-2*p8-p9)) >> 3;
            img_sobel.at<uint8_t>(i,j) = G;
        }
    }
}

void *sobel_thread1(void *arg){
    grayscale_442(0, 269, 0, 1919);
    sobel_442(1, 269, 1, 1918);
    //pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread2(void *arg){
    grayscale_442(270, 539, 0, 1919);
    sobel_442(270, 539, 1, 1918);
    //pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread3(void *arg){
    grayscale_442(540, 809, 0, 1919);
    sobel_442(540, 809, 1, 1918);
    //pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread4(void *arg){
    grayscale_442(810, 1079, 0, 1919);
    sobel_442(810, 1078, 1, 1918);
    //pthread_barrier_wait(&barrier);
    return NULL;
}

int main(int argc, char** argv){

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    VideoCapture cap(argv[1]);

    while(true){

        // Get video frame, break while loop if no more frames
        cap.read(img_color);
        if (img_color.empty()){
            break;
	    }

        // Run threads
        pthread_t thread1;
        pthread_t thread2;
        pthread_t thread3;
        pthread_t thread4;

        //pthread_barrier_init(&barrier, NULL, 4);

        pthread_create(&thread1, NULL, sobel_thread1, NULL);
        pthread_create(&thread2, NULL, sobel_thread2, NULL);
        pthread_create(&thread3, NULL, sobel_thread3, NULL);
        pthread_create(&thread4, NULL, sobel_thread4, NULL);

        pthread_join(thread1, NULL);
        pthread_join(thread2, NULL);
        pthread_join(thread3, NULL);
        pthread_join(thread4, NULL);

        //pthread_barrier_destroy(&barrier);

        // Display image windows
        namedWindow("Sobel", 0);
        namedWindow("Color", 0);
        resizeWindow("Sobel",img_sobel.cols,img_sobel.rows);
        resizeWindow("Color",500,300);
        moveWindow("Sobel", 0,0);
        moveWindow("Color", 0,0);
        imshow("Sobel", img_sobel);
        imshow("Color", img_color);

        // Stop program is "ESC" is pressed
        char c=(char)waitKey(10);
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
