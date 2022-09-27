#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
 // Read the image file
 Mat image = imread(argv[1],1);

 // Check for failure
 if (image.empty()) 
 {
  cout << "Could not open or find the image" << endl;
  cin.get(); //wait for any key press
  return -1;
 }

 String windowName = "Hawk"; //Name of the window
 namedWindow(windowName,0); // Create a window
 resizeWindow("Hawk",500,500); // Resize window
 imshow(windowName, image); // Show our image inside the created window.
 waitKey(0); // Wait for any keystroke in the window
 destroyWindow(windowName); //destroy the created window

 return 0;
}
