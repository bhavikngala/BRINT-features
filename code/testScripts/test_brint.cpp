#include "../utils/misc.hpp"
#include "../utils/brint.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	Mat image = imread("./data/leaf.png", IMREAD_GRAYSCALE);
	Mat brintImage;
	Mat hist;

//	Mat saltpepper_noise = Mat::zeros(image.rows, image.cols,CV_8U);
//	randu(saltpepper_noise,0,255);
//
//	Mat black = saltpepper_noise < 30;
//	Mat white = saltpepper_noise > 225;

//	Mat saltpepper_img = image.clone();
//	saltpepper_img.setTo(255,white);
//	saltpepper_img.setTo(0,black);
//	imwrite("./results/leaf_sp.jpg", saltpepper_img);

//	namedWindow("saltNpepper", WINDOW_NORMAL);
//	resizeWindow("saltNpepper", 1080, 1280);
//	imshow("saltNpepper", saltpepper_img);

	features::Brint::brint_s(image, brintImage, hist, 2, 32);
	imwrite("./results/as_brint_s_r2_n32.jpg", brintImage);

//	namedWindow( "Image", WINDOW_NORMAL);
//	resizeWindow("Image", 1280, 1080);
//	namedWindow( "Texture r=2, n=32", WINDOW_NORMAL );
//	resizeWindow("Texture r=2, n=32", 1280, 1080);
//	imshow("Image", image);
//	imshow( "Texture r=2, n=32", brintImage );


//	namedWindow( "Texture r=4, n=32", WINDOW_NORMAL );
//	resizeWindow("Texture r=4, n=32", 1280, 1080);
	features::Brint::brint_s(image, brintImage, hist, 4, 32);
	imwrite("./results/as_brint_s_r4_n32.jpg", brintImage);
//	imshow( "Texture r=4, n=32", brintImage );

//	namedWindow( "Texture r=8, n=32", WINDOW_NORMAL );
//	resizeWindow("Texture r=8, n=32", 1280, 1080);
	features::Brint::brint_s(image, brintImage, hist, 8, 32);
	imwrite("./results/as_brint_s_r8_n32.jpg", brintImage);
//	imshow( "Texture r=8, n=32", brintImage );

//	namedWindow( "Texture r=12, n=32", WINDOW_NORMAL );
//	resizeWindow("Texture r=12, n=32", 1280, 1080);
	features::Brint::brint_s(image, brintImage, hist, 12, 32);
	imwrite("./results/as_brint_s_r12_n32.jpg", brintImage);
//	imshow( "Texture r=12, n=32", brintImage );

//    waitKey(0);
	return 0;

}
