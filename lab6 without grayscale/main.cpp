#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>

#define BLUE 0b00010010
#define GREEN 0b10110111
#define RED 0b00110110


using namespace cv;

pthread_barrier_t barrier;

Mat img_color(1080, 1920, CV_8UC3);
Mat img_sobel(1080, 1920, CV_8UC1);

/*void grayscale_442(int row_start, int row_end, int col_start, int col_end){

	uint16x8_t blue, green, red, wide_gray;

	uint8_t *BGR_ptr = &img_color.at<Vec3b>(row_start, col_start)[0];  //points the beginning of 8 pixel row
	uint8_t *Gray_ptr = &img_gray.at<uint8_t>(row_start, col_start);

	uint8x8x3_t BGR;
	uint8x8_t gray;

    for (int row = row_start; row <= row_end; row++){
	for (int col = col_start; col <= col_end; col = col + 8){

	    BGR_ptr = BGR_ptr + 24;
	    Gray_ptr = Gray_ptr + 8;

	    BGR = vld3_u8(BGR_ptr);	//loads 8 RGB pixles into 3 8x8x3 vectors

	    blue = vmovl_u8(BGR.val[0]);  //widens vectors from 8x8 to 16x8
	    green = vmovl_u8(BGR.val[1]);
	    red = vmovl_u8(BGR.val[2]);


        wide_gray = vaddq_u16(vaddq_u16(vmulq_n_u16(blue, BLUE),  //perfoms RGB algorithm on pixels
                                     vmulq_n_u16(green, GREEN)),
                                     vmulq_n_u16(red, RED));

	    wide_gray = vshrq_n_u16(wide_gray, 8);
	    gray = vqmovn_u16(wide_gray);

	    vst1_u8(Gray_ptr, gray);

		}
	}
}*/

uint8x8_t grayscale(uint8_t *BGR_ptr){ //converts colored uint8x8x3_t vector at pointer to grayscale uint8x8_t

	uint16x8_t blue, green, red, wide_gray;
	uint8x8x3_t BGR;

    BGR = vld3_u8(BGR_ptr);	//loads 8 RGB pixles into 3 8x8x3 vectors

    blue = vmovl_u8(BGR.val[0]);  //widens vectors from 8x8 to 16x8
    green = vmovl_u8(BGR.val[1]);
    red = vmovl_u8(BGR.val[2]);


    wide_gray = vaddq_u16(vaddq_u16(vmulq_n_u16(blue, BLUE),  //perfoms RGB algorithm on pixels
                                    vmulq_n_u16(green, GREEN)),
                                    vmulq_n_u16(red, RED));

    wide_gray = vshrq_n_u16(wide_gray, 8);
    return vqmovn_u16(wide_gray);

}

void sobel_442(int row_start, int row_end, int col_start, int col_end){

    uint8x8_t p1, p2, p3, p4, p5, p6, p7, p8, p9, g; //byte sized vectors
    int8x8_t p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s, p9s, gs; //signed byte vectors
    int16x8_t P1, P2, P3, P4, P5, P6, P7, P8, P9, G, T1, T2, T3, T4, T5, T6, T7, T8; //16 bit sized vectors
    uint8_t *p1p,*p2p,*p3p,*p4p,*p5p,*p6p,*p7p,*p8p,*p9p,*sobel_ptr; //pointers

    p1p = &img_color.at<Vec3b>(row_start-1,col_start-1)[0];
    p2p = &img_color.at<Vec3b>(row_start-1,col_start)[0];
    p3p = &img_color.at<Vec3b>(row_start-1,col_start+1)[0];
    p4p = &img_color.at<Vec3b>(row_start-1,col_start+1)[0];
    p5p = &img_color.at<Vec3b>(row_start,col_start)[0];
    p6p = &img_color.at<Vec3b>(row_start,col_start+1)[0];
    p7p = &img_color.at<Vec3b>(row_start+1,col_start-1)[0];
    p8p = &img_color.at<Vec3b>(row_start+1,col_start)[0];
    p9p = &img_color.at<Vec3b>(row_start+1,col_start+1)[0];
    sobel_ptr = &img_sobel.at<uint8_t>(row_start,col_start);

    // (0,0) is top left, increasing down and to the right
    for(int i=row_start; i<=row_end; i++){
        for(int j=col_start; j<=col_end; j = j + 8){

	    p1p = p1p + 24;
	    p2p = p2p + 24;
	    p3p = p3p + 24;
	    p4p = p4p + 24;
	    p5p = p5p + 24;
	    p6p = p6p + 24;
	    p7p = p7p + 24;
	    p8p = p8p + 24;
	    p9p = p9p + 24;
	    sobel_ptr = sobel_ptr + 8;

        //converts colored uint8x8x3_t vector at pointer to grayscale uint8x8_t

	    p1 = grayscale(p1p);
	    p2 = grayscale(p2p);
	    p3 = grayscale(p3p);
	    p4 = grayscale(p4p);
	    p5 = grayscale(p5p);
	    p6 = grayscale(p6p);
	    p7 = grayscale(p7p);
	    p8 = grayscale(p8p);
	    p9 = grayscale(p9p);

	    //loads in grayscale vectors from memory address
	    p1 = vld1_u8(p1p);
	    p2 = vld1_u8(p2p);
	    p3 = vld1_u8(p3p);
	    p4 = vld1_u8(p4p);
	    p5 = vld1_u8(p5p);
	    p6 = vld1_u8(p6p);
	    p7 = vld1_u8(p7p);
	    p8 = vld1_u8(p8p);
	    p9 = vld1_u8(p9p);

	    //casts unsigned bytes in vectors to signed bytes
	    p1s = vreinterpret_s8_u8(p1);
	    p2s = vreinterpret_s8_u8(p2);
	    p3s = vreinterpret_s8_u8(p3);
	    p4s = vreinterpret_s8_u8(p4);
	    p5s = vreinterpret_s8_u8(p5);
	    p6s = vreinterpret_s8_u8(p6);
	    p7s = vreinterpret_s8_u8(p7);
	    p8s = vreinterpret_s8_u8(p8);
	    p9s = vreinterpret_s8_u8(p9);

	    //widens the vectors to be operated with
	    P1 = vmovl_s8(p1s);
	    P2 = vmovl_s8(p2s);
	    P3 = vmovl_s8(p3s);
	    P4 = vmovl_s8(p4s);
	    P5 = vmovl_s8(p5s);
	    P6 = vmovl_s8(p6s);
	    P7 = vmovl_s8(p7s);
	    P8 = vmovl_s8(p8s);
	    P9 = vmovl_s8(p9s);

	    T1 = vsubq_s16(P3, P1);
	    T2 = vsubq_s16(vmulq_n_s16(P6, 2), vmulq_n_s16(P4, 2));
	    T3 = vsubq_s16(P9, P7);

	    T4 = vsubq_s16(P1, P7);
	    T5 = vsubq_s16(vmulq_n_s16(P2, 2), vmulq_n_s16(P8, 2));
	    T6 = vsubq_s16(P3, P9);

	    T7 = vaddq_s16(T1, vaddq_s16(T2, T3));
	    T8 = vaddq_s16(T4, vaddq_s16(T5, T6));

	    G = vshrq_n_s16(vaddq_s16(vabsq_s16(T7), vabsq_s16(T8)), 3);

	    gs = vmovn_s16(G);
	    g = vreinterpret_u8_s8(gs);

	    vst1_u8(sobel_ptr, g);
        }
    }
}

void *sobel_thread1(void *arg){
    sobel_442(1, 269, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread2(void *arg){
    sobel_442(270, 539, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread3(void *arg){
    sobel_442(540, 809, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread4(void *arg){
    sobel_442(810, 1078, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}

int main(int argc, char** argv){

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    VideoCapture cap(argv[1]);

    pthread_t thread1;
    pthread_t thread2;
    pthread_t thread3;
    pthread_t thread4;

    while(true){

        // Get video frame, break while loop if no more frames
        cap.read(img_color);
        if (img_color.empty()){
            break;
	    }

        // Run threads

        pthread_barrier_init(&barrier, NULL, 4);

        pthread_create(&thread1, NULL, sobel_thread1, NULL);
        pthread_create(&thread2, NULL, sobel_thread2, NULL);
        pthread_create(&thread3, NULL, sobel_thread3, NULL);
        pthread_create(&thread4, NULL, sobel_thread4, NULL);

        pthread_barrier_destroy(&barrier);

        // Display image windows
        namedWindow("Sobel", 0);
        namedWindow("Color", 0);
        resizeWindow("Sobel",img_sobel.cols,img_sobel.rows);
        resizeWindow("Color",500,300);
        moveWindow("Sobel", 0,0);
        moveWindow("Color", 0,0);
        imshow("Sobel", img_sobel);
        imshow("Color", img_color);


        // Stop program if "ESC" is pressed
        char c=(char)waitKey(10);
        if(c==27){
            break;
        }
    }

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    pthread_join(thread4, NULL);

    // Display timing results
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\nExecution Time: " << duration.count() << " ms\n\n";

	return 0;
}
