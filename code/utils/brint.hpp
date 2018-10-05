#ifndef BRINT_HPP
#define  BRINT_HPP

#include <cmath>
#include <limits>
#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "misc.hpp"

using namespace cv;
using namespace std;

namespace features{
	class Brint{
	public:
		static void brint_s(const Mat& src, Mat& dst,
				Mat& hist, int radius, int neighbours, bool normalizeHist);

		static void brint_m(const Mat& src, Mat& dst,
				Mat& hist, int radius, int neighbours, bool normalizeHist);

		static void brint_c(const Mat& src, Mat& dst,
				Mat& hist, int radius, int neighbours, bool normalizeHist);

		static void brint_cs_cm(const Mat& src, Mat& hist,
				int radius, int neighbours, bool normalizeHist);
	};
}

#endif
