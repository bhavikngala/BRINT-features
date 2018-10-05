#ifndef MISC_HPP
#define MISC_HPP

#include <vector>
#include <array>
#include <climits>
#include <cmath>

using std::vector;
using std::array;

namespace misc{

	// find min{ROR(x, i)} i=0..i; ROR -> right circular shift
	unsigned char minROR(unsigned char x, int numShifts);

	// vector of cordinates of circular neighbourhood
	// and interpolation weights
	// [floor_x, floor_y, ceil_x, ceil_y, w1, w2, w3, w4]
	vector<array<float, 8>> getNeighbourhoodCoordinates(int radius,
		int neighbours);
	
}

#endif
