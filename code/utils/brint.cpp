#include "brint.hpp"
#include <iostream>

/*
** Author: Bhavik N Gala
** Email: bhavikna@buffalo.edu
*/

namespace features{

	void Brint::brint_s(const Mat& src, Mat& dst, Mat& hist, int radius,
		int neighbours, bool normalizeHist){

		// computes 'Binary Noise Tolerant Sign'
		// features for an given image

		// TODO: assert neighbours%8 == 0
		int q = (int)(neighbours/8);

		// TODO: check out other interpolation methods
		// interpolating pixel values
		// vector of cordinates and interpolation weights
		// [floor_x, floor_y, ceil_x, ceil_y, w1, w2, w3, w4]
		vector<array<float, 8>> neighbourhoodCoords = \
			misc::getNeighbourhoodCoordinates(radius, neighbours);

		// int dataType = src.depth();
		dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8U);

		// iterating through each pixel value
		for(int y=radius; y<src.rows-radius; y++){
			for(int x=radius; x<src.cols-radius; x++){
				// getting the neighbourhood
				Mat neighbourhood(src, Rect(x-radius, y-radius,
					2*radius+1, 2*radius+1));
				uint8_t *nData = neighbourhood.data;

				// array of circular neighbourhood pixel values
				float neighbourVector[neighbours];

				// iterating through all the points in the circular
				// neighbourhood and interpolating them
				for(int p=0; p<neighbours; p++){

					array<float, 8> xy = neighbourhoodCoords[p];

					// s = w1*src(fy,fx)+w2*src(fy,cx)+w3*src(cy,fx)+w4*src(cy,cx)
					Scalar p1 = neighbourhood.at<uchar>((int)(xy[1]), (int)(xy[0]));
					Scalar p2 = neighbourhood.at<uchar>((int)(xy[1]), (int)(xy[2]));
					Scalar p3 = neighbourhood.at<uchar>((int)(xy[3]), (int)(xy[0]));
					Scalar p4 = neighbourhood.at<uchar>((int)(xy[3]), (int)(xy[2]));
					//neighbourVector[p] = xy[4]*nData[(int)(xy[1]), (int)(xy[0])] + xy[5]*nData[(int)(xy[1]), (int)(xy[2])] + xy[6]*nData[(int)(xy[3]), (int)(xy[0])] + xy[7]*nData[(int)(xy[3]), (int)(xy[2])];
					neighbourVector[p] = xy[4]*p1.val[0] + xy[5]*p2.val[0] + xy[6]*p3.val[0] + xy[7]*p4.val[0];
				}

				// transform the neighbour vector by local averaging
				// along the arc
				unsigned char bnt_s = 0;
				for(int i=0; i<8; i++){
					float y = 0;
					// y_r,q,i=(1/q)*sum(from k=0,..,q-1)(x_r,8q,qi+k)
					for(int k=0; k<q; k++){
						y += neighbourVector[q*i + k];
					}
					y /= q;

					// compute LBP
					unsigned char s = ((y-nData[radius, radius])>=0) ? 1 : 0;
					bnt_s += s * (unsigned char)(pow(2, i));
				}

				// addding rotation invariance
				// BRINT_S = min{ROR(BNT_S, i)} i=0..7
				dst.at<unsigned char>(y-radius, x-radius) = misc::minROR(bnt_s, q);
			}
		}

		// compute histogram
		// number of bins in the histogram
		int histSize = 256;
		// set the range of values, from 0 to 255
		// upper value is exclusive
		float range[] = {0, 2};
		const float* histRange = {range};
		// uniform size of bins
		bool uniform = true;
		// set this flag to accumulate values of bins with previous histogram
		bool accumulate = false;
		cv::calcHist(&dst, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		if(normalizeHist){
			// min output value = 0; max output value = 255
			normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		}
	}

	void Brint::brint_m(const Mat& src, Mat& dst, Mat& hist, int radius,
		int neighbours, bool normalizeHist){
		// computes 'Binary Noise Tolerant Magnitude'
		// features for an given image

		// TODO: assert neighbours%8 == 0
		int q = (int)(neighbours/8);

		// TODO: check out other interpolation methods
		// interpolating pixel values
		// vector of cordinates and interpolation weights
		// [floor_x, floor_y, ceil_x, ceil_y, w1, w2, w3, w4]
		vector<array<float, 8>> neighbourhoodCoords = \
			misc::getNeighbourhoodCoordinates(radius, neighbours);

		dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8U);

		// iterating through each pixel value
		for(int y=radius; y<src.rows-radius; y++){
			for(int x=radius; x<src.cols-radius; x++){
				// getting the neighbourhood
				Mat neighbourhood(src, Rect(x-radius, y-radius,
					2*radius+1, 2*radius+1));
				uint8_t *nData = neighbourhood.data;

				// array of circular neighbourhood pixel values
				float neighbourVector[neighbours];

				// iterating through all the points in the circular
				// neighbourhood and interpolating them
				for(int p=0; p<neighbours; p++){

					array<float, 8> xy = neighbourhoodCoords[p];

					// s = w1*src(fy,fx)+w2*src(fy,cx)+w3*src(cy,fx)+w4*src(cy,cx)
					neighbourVector[p] = \
					xy[4]*nData[(int)(xy[1]), (int)(xy[0])] + \
					xy[5]*nData[(int)(xy[1]), (int)(xy[2])] + \
					xy[6]*nData[(int)(xy[3]), (int)(xy[0])] + \
					xy[7]*nData[(int)(xy[3]), (int)(xy[2])];

					// delta_r = abs(x_r,p,i - x_c)
					neighbourVector[p] = abs(neighbourVector[p] - nData[radius, radius]);
				}

				// transform the neighbour vector by local averaging
				// along the arc
				float z[8];
				for(int i=0; i<8; i++){
					float y = 0;
					// y_r,q,i=(1/q)*sum(from k=0,..,q-1)(x_r,8q,qi+k)
					for(int k=0; k<q; k++){
						y += neighbourVector[q*i + k];
					}
					y /= q;
					z[i] = y;
				}
				// compute mean of the array z
				float mu_z = 0;
				for(int i=0; i<8; i++){
					mu_z += z[i];
				}
				mu_z /= 8;

				unsigned char bnt_m = 0;
				for(int i=0; i<8; i++){
					// compute LBP
					unsigned char s = ((z[i]-mu_z)>=0) ? 1 : 0;
					bnt_m += s * (unsigned char)(pow(2, i));
				}

				// addding rotation invariance
				// BRINT_M = min{ROR(BNT_M, i)} i=0..7
				dst.at<unsigned char>(y-radius, x-radius) = misc::minROR(bnt_m, q);
			}
		}

		// compute histogram
		// number of bins in the histogram
		int histSize = 256;
		// set the range of values, from 0 to 255
		// upper value is exclusive
		float range[] = {0, 2};
		const float* histRange = {range};
		// uniform size of bins
		bool uniform = true;
		// set this flag to accumulate values of bins with previous histogram
		bool accumulate = false;
		calcHist(&dst, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		if(normalizeHist){
			// min output value = 0; max output value = 255
			normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		}
	}

	void Brint::brint_c(const Mat& src, Mat& dst, Mat& hist, int radius,
		int neighbours, bool normalizeHist){
		// computes 'Binary Noise Tolerant Center'
		// features for an given image

		dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8U);

		uint8_t *srcData = src.data;
		// mean of all the pixels in the image
		float mu = (cv::mean(src(Range(radius, src.rows-radius),
						  Range(radius, src.cols-radius))))[0];

		// iterating through each pixel value
		for(int y=radius; y<src.rows-radius; y++){
			for(int x=radius; x<src.cols-radius; x++){
				unsigned char s = ((srcData[y, x]-mu)>=0) ? 1 : 0;
				dst.at<unsigned char>(y-radius, x-radius) = s;
			}
		}

		// compute histogram
		// number of bins in the histogram
		int histSize = 2;
		// set the range of values, from 0 to 1
		// upper value is exclusive
		float range[] = {0, 2};
		const float* histRange = {range};
		// uniform size of bins
		bool uniform = true;
		// set this flag to accumulate values of bins with previous histogram
		bool accumulate = false;
		calcHist(&dst, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		if(normalizeHist){
			// min output value = 0; max output value = 255
			normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		}
	}

	void Brint::brint_cs_cm(const Mat& src, Mat& hist, int radius, int neighbours, bool normalizeHist){
		// computes BRINT feature
		// computes joint histogram from BNT_C a)nd BNT_S; BNT_C and BNT_M
		// concatenates the two joint histogram

		// TODO: assert neighbours%8 == 0
		int q = (int)(neighbours/8);
		// TODO: check out other interpolation methods
		// interpolating pixel values
		// vector of cordinates and interpolation weights
		// [floor_x, floor_y, ceil_x, ceil_y, w1, w2, w3, w4]
		vector<array<float, 8>> neighbourhoodCoords = \
			misc::getNeighbourhoodCoordinates(radius, neighbours);

		Mat bnt_s = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8U);
		Mat bnt_m = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8U);
		Mat bnt_c = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8U);

		uint8_t *srcData = src.data;

		// mean of all the pixels in the image
		float mu = (cv::mean(src(Range(radius, src.rows-radius),
								 Range(radius, src.cols-radius))))[0];

		for(int y=radius; y<src.rows-radius; y++){
			for(int x=radius; x<src.cols-radius; x++){
				// getting the neighbourhood
				Mat neighbourhood(src, Rect(x-radius, y-radius,
											2*radius+1, 2*radius+1));
				uint8_t *nData = neighbourhood.data;

				// array of circular neighbourhood pixel values
				float neighbourVector[neighbours];

				// iterating through all the points in the circular
				// neighbourhood and interpolating them
				for(int p=0; p<neighbours; p++){

					array<float, 8> xy = neighbourhoodCoords[p];

					// s = w1*src(fy,fx)+w2*src(fy,cx)+w3*src(cy,fx)+w4*src(cy,cx)
					Scalar p1 = neighbourhood.at<uchar>((int)(xy[1]), (int)(xy[0]));
					Scalar p2 = neighbourhood.at<uchar>((int)(xy[1]), (int)(xy[2]));
					Scalar p3 = neighbourhood.at<uchar>((int)(xy[3]), (int)(xy[0]));
					Scalar p4 = neighbourhood.at<uchar>((int)(xy[3]), (int)(xy[2]));
					neighbourVector[p] = xy[4]*p1.val[0] + xy[5]*p2.val[0] + xy[6]*p3.val[0] + xy[7]*p4.val[0];
				}

				// transform the neighbour vector by local averaging
				// along the arc
				float z[8];
				for(int i=0; i<8; i++){
					float y = 0;
					// y_r,q,i=(1/q)*sum(from k=0,..,q-1)(x_r,8q,qi+k)
					for(int k=0; k<q; k++){
						y += neighbourVector[q*i + k];
					}
					y /= q;
					z[i] = y;
				}
				// compute mean of the array z
				float mu_z = 0;
				for(int i=0; i<8; i++){
					mu_z += z[i];
				}
				mu_z /= 8;

				unsigned char bnt_s1 = 0;
				unsigned char bnt_m1 = 0;
				unsigned char s;
				for(int i=0; i<8; i++){
					// compute LBP
					// bnt_s
					s = ((z[i]-nData[radius, radius])>=0) ? 1 : 0;
					bnt_s1 += s * (unsigned char)(pow(2, i));
					// bnt_m
					s = ((z[i]-mu_z)>=0) ? 1 : 0;
					bnt_m1 += s * (unsigned char)(pow(2, i));
				}
				// bnt_c
				s = ((srcData[y, x]-mu)>=0) ? 1 : 0;

				bnt_s.at<unsigned char>(y-radius, x-radius) = misc::minROR(bnt_s1, q);
				bnt_m.at<unsigned char>(y-radius, x-radius) = misc::minROR(bnt_m1, q);
				bnt_c.at<unsigned char>(y-radius, x-radius) = s;
			}
		}

		// compute histogram
		// number of bins in the histogram
		int histSize1 = 256;
		int histSize2 = 2;
		// set the range of values, from 0 to 255
		// upper value is exclusive
		float range1[] = {0, 256};
//		float range2[] = {0, 2};
		const float* histRange1 = {range1};
//		const float* histRange2= {range2};

		Mat hist_cs;
		Mat hist_cm;

		calcHist(&bnt_s, 1, 0, Mat(), hist_cs, 1, &histSize1, &histRange1, true, false); // clear the hist_cs array
		calcHist(&bnt_c, 1, 0, Mat(), hist_cs, 1, &histSize1, &histRange1, true, true); // accumulate the hist_cs array

		calcHist(&bnt_m, 1, 0, Mat(), hist_cm, 1, &histSize1, &histRange1, true, false); // clear the hist_cs array
		calcHist(&bnt_c, 1, 0, Mat(), hist_cm, 1, &histSize1, &histRange1, true, true); // accumulate the hist_cs array

		if(normalizeHist){
			// min output value = 0; max output value = 255
			normalize(hist_cs, hist_cs, 0, 255, NORM_MINMAX, -1, Mat());
			normalize(hist_cm, hist_cm, 0, 255, NORM_MINMAX, -1, Mat());
		}

		// concatenate the histograms
		vconcat(hist_cs, hist_cm, hist);
	}

}
