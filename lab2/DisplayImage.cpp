#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(	int argc,
			char** argv
			){

	// Read the image file
	Mat image = imread(argv[1],1);

	// Check for failure
	if (image.empty()){
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	//Name the window
	String windowName = "Hawk";
 
	// Create a window
	namedWindow(windowName,0);
 
	// Resize the window
	resizeWindow("Hawk",500,500);
 
	// Show image inside the window
	imshow(windowName, image);
 
	// Wait for any keystroke, then destroy the created window
	waitKey(0);
	destroyWindow(windowName);

	return 0;
}
