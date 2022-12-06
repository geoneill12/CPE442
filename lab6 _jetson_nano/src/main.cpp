#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>

using namespace cv;

pthread_barrier_t barrier;

Mat img_color(1080, 1920, CV_8UC3);
Mat img_gray(1080, 1920, CV_8UC1);
Mat img_sobel(1080, 1920, CV_8UC1);
int *ptr_c_i = img_color.ptr<int>(0,0);
int *ptr_g_i = img_gray.ptr<int>(0,0);
int *ptr_s_i = img_sobel.ptr<int>(0,0);

void grayscale_442(int row_start, int row_end, int col_start, int col_end){

	uint8_t *ptr_c = (uint8_t*)ptr_c_i;
	uint8_t *ptr_g = (uint8_t*)ptr_g_i;

	if(row_start == 270){
		ptr_c += 1555200;
		ptr_g += 518400;
	}
	if(row_start == 540){
		ptr_c += 3110400;
		ptr_g += 1036800;
	}
	if(row_start == 810){
		ptr_c += 4665600;
		ptr_g += 1555200;
	}

	uint8x16x3_t BGR;
	uint8x16_t GRAY;

    for (int row = row_start; row <= row_end; row++){
		for (int j=1; j<=240; j+=2){

	    	BGR = vld3q_u8(ptr_c);

			BGR.val[0] = vshrq_n_u8(BGR.val[0], 4);
			BGR.val[1] = vshrq_n_u8(BGR.val[1], 2);
			BGR.val[2] = vshrq_n_u8(BGR.val[2], 4);

			GRAY = vaddq_u8(BGR.val[0], BGR.val[1]);
			GRAY = vaddq_u8(GRAY, BGR.val[1]);
			GRAY = vaddq_u8(GRAY, BGR.val[1]);
			GRAY = vaddq_u8(GRAY, BGR.val[2]);
			GRAY = vaddq_u8(GRAY, BGR.val[2]);
			GRAY = vaddq_u8(GRAY, BGR.val[2]);

	    	vst1q_u8(ptr_g, GRAY);

	    	ptr_c += 48;
	    	ptr_g += 16;
		}
	}
}

void sobel_442(int row_start, int row_end, int col_start, int col_end){

	int FLAG = 0;
	uint8_t TEMP[8];
	uint8_t *ptr_TEMP = TEMP;

	uint8_t *ptr_g = (uint8_t*)ptr_g_i;
	uint8_t *ptr_s = (uint8_t*)ptr_s_i;

	if(row_start == 1){
		ptr_s += 1921;
	}
	if(row_start == 270){
		ptr_g += 518400;
		ptr_s += 520321;
	}
	if(row_start == 540){
		ptr_g += 1036800;
		ptr_s += 1038721;
	}
	if(row_start == 810){
		ptr_g += 1555200;
		ptr_s += 1557121;
	}

	uint8x8x3_t p123_64b, p456_64b, p789_64b;
	uint16x8_t p1u, p2u, p3u, p4u, p6u, p7u, p8u, p9u;
	int16x8_t p1s, p2s, p3s, p4s, p6s, p7s, p8s, p9s;
	int16x8_t Gx, Gy, G;
	int8x8_t G_64b_s;
	uint8x8_t G_64b;

    for(int i=row_start; i<=row_end; i++){
        for(int j=1; j<=238; j++){

			p123_64b = vld3_u8(ptr_g);
			p456_64b = vld3_u8(ptr_g+1920);
			p789_64b = vld3_u8(ptr_g+3840);

			p1u = vmovl_u8(p123_64b.val[0]);
			p2u = vmovl_u8(p123_64b.val[1]);
			p3u = vmovl_u8(p123_64b.val[2]);
			p4u = vmovl_u8(p456_64b.val[0]);
			p6u = vmovl_u8(p456_64b.val[2]);
			p7u = vmovl_u8(p789_64b.val[0]);
			p8u = vmovl_u8(p789_64b.val[1]);
			p9u = vmovl_u8(p789_64b.val[2]);

			p1s = vreinterpretq_s16_u16(p1u);
			p2s = vreinterpretq_s16_u16(p2u);
			p3s = vreinterpretq_s16_u16(p3u);
			p4s = vreinterpretq_s16_u16(p4u);
			p6s = vreinterpretq_s16_u16(p6u);
			p7s = vreinterpretq_s16_u16(p7u);
			p8s = vreinterpretq_s16_u16(p8u);
			p9s = vreinterpretq_s16_u16(p9u);

			Gx = vaddq_s16(p3s, p9s);
			Gx = vaddq_s16(Gx, p6s);
			Gx = vaddq_s16(Gx, p6s);
			Gx = vsubq_s16(Gx, p1s);
			Gx = vsubq_s16(Gx, p7s);
			Gx = vsubq_s16(Gx, p4s);
			Gx = vsubq_s16(Gx, p4s);

			Gy = vaddq_s16(p1s, p3s);
			Gy = vaddq_s16(Gy, p2s);
			Gy = vaddq_s16(Gy, p2s);
			Gy = vsubq_s16(Gy, p7s);
			Gy = vsubq_s16(Gy, p9s);
			Gy = vsubq_s16(Gy, p8s);
			Gy = vsubq_s16(Gy, p8s);

			Gx = vabsq_s16(Gx);
			Gy = vabsq_s16(Gy);

			G = vaddq_s16(Gx, Gy);
			G_64b_s = vmovn_s16(G);
			G_64b = vreinterpret_u8_s8(G_64b_s);

	    	vst1_u8(ptr_TEMP, G_64b);

			*(ptr_s+0) = TEMP[0];
			*(ptr_s+3) = TEMP[1];
			*(ptr_s+6) = TEMP[2];
			*(ptr_s+9) = TEMP[3];
			*(ptr_s+12) = TEMP[4];
			*(ptr_s+15) = TEMP[5];
			*(ptr_s+18) = TEMP[6];
			*(ptr_s+21) = TEMP[7];

			if(j == 238){
				ptr_g += 24;
				ptr_s += 24;
				FLAG = 0;
				continue;
			}
			if(FLAG == 2){
				ptr_g += 22;
				ptr_s += 22;
				FLAG = 0;
			}
			if(FLAG == 1){
				ptr_g += 1;
				ptr_s += 1;
				FLAG += 1;
			}
			if(FLAG == 0){
				ptr_g += 1;
				ptr_s += 1;
				FLAG += 1;
			}
        }
    }
}

void grayscale_442_m(int row_start, int row_end, int col_start, int col_end){

	int p0, p1, p2;

	uint8_t *ptr_c = (uint8_t*)ptr_c_i;
	uint8_t *ptr_g = (uint8_t*)ptr_g_i;

	if(row_start == 270){
		ptr_c += 1555200;
		ptr_g += 518400;
	}
	if(row_start == 540){
		ptr_c += 3110400;
		ptr_g += 1036800;
	}
	if(row_start == 810){
		ptr_c += 4665600;
		ptr_g += 1555200;
	}
    
    for (int row = row_start; row <= row_end; row++){
		for (int col = col_start; col <= col_end; col++){
            p0 = (*(ptr_c + 0) >> 4);     	//Blue
            p1 = (*(ptr_c + 1) >> 2)*3;		//Green
            p2 = (*(ptr_c + 2) >> 4)*3;     //Red
            *ptr_g = p0+p1+p2;
			ptr_c += 3;
			ptr_g += 1;
		}
	}
}

void sobel_442_m(int row_start, int row_end, int col_start, int col_end){

    int p1,p2,p3,p4,p5,p6,p7,p8,p9,G;

	uint8_t *ptr_g = (uint8_t*)ptr_g_i;
	uint8_t *ptr_s = (uint8_t*)ptr_s_i;

	if(row_start == 1){
		ptr_g += 1921;
		ptr_s += 1921;
	}
	if(row_start == 270){
		ptr_g += 518401;
		ptr_s += 518401;
	}
	if(row_start == 540){
		ptr_g += 1036801;
		ptr_s += 1036801;
	}
	if(row_start == 810){
		ptr_g += 1555201;
		ptr_s += 1555201;
	}

    // (0,0) is top left, increasing down and to the right
    for(int i=row_start; i<=row_end; i++){
        for(int j=col_start; j<=col_end; j++){
            if(j == 1){
                p1 = *(ptr_g-1921);
                p2 = *(ptr_g-1920);
                p3 = *(ptr_g-1919);
                p4 = *(ptr_g-1);
                p5 = *(ptr_g);
                p6 = *(ptr_g+1);
                p7 = *(ptr_g+1919);
                p8 = *(ptr_g+1920);
                p9 = *(ptr_g+1921);
            }
            if(j != 1){
                p1 = p2;
                p2 = p3;
                p3 = *(ptr_g-1919);
                p4 = p5;
                p5 = p6;
                p6 = *(ptr_g+1);
                p7 = p8;
                p8 = p9;
                p9 = *(ptr_g+1921);
            }
            G = (abs(-p1-2*p4-p7+p3+2*p6+p9) + abs(p1+2*p2+p3-p7-2*p8-p9)) >> 3;
            *ptr_s = G;

			if(j == 1918){
				ptr_g += 3;
				ptr_s += 3;
			}
			if(j != 1918){
				ptr_g += 1;
				ptr_s += 1;
			}
        }
    }
}

void *sobel_thread1(void *arg){
    grayscale_442(0, 269, 0, 1919);
    sobel_442(1, 269, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread2(void *arg){
    grayscale_442(270, 539, 0, 1919);
    sobel_442(270, 539, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread3(void *arg){
    grayscale_442(540, 809, 0, 1919);
    sobel_442(540, 809, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}
void *sobel_thread4(void *arg){
    grayscale_442(810, 1079, 0, 1919);
    sobel_442(810, 1078, 1, 1918);
    pthread_barrier_wait(&barrier);
    return NULL;
}

int main(int argc, char** argv){

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    VideoCapture cap(argv[1]);

	namedWindow("Sobel", 0);
    namedWindow("Color", 0);
    //namedWindow("Gray", 0);
    resizeWindow("Sobel",500,300);
    resizeWindow("Color",500,300);
    //resizeWindow("Gray",500,300);
    moveWindow("Sobel", 1000,0);
    moveWindow("Color", 0,0);
    //moveWindow("Gray", 0, 500);

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
        imshow("Sobel", img_sobel);
        imshow("Color", img_color);
        //imshow("Gray", img_gray);


        // Stop program if "ESC" is pressed
        char c=(char)waitKey(1);
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
