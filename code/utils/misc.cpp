#include "misc.hpp"

/*
** Author: Bhavik N Gala
** Email: bhavikna@buffalo.edu
*/

using std::floor;
using std::ceil;
using std::min;

namespace misc{
	vector<array<float, 8>> getNeighbourhoodCoordinates(int radius,
		int neighbours){
		// vector of cordinates and interpolation weights
		// [floor_x, floor_y, ceil_x, ceil_y, w1, w2, w3, w4]
		vector<array<float, 8>> neighbourhoodCoords;
		//float theta = 2.0*M_PI/neighbours;

		for(int i=0; i<neighbours; i++){
			// array template
			array<float, 8> neighbourhoodCoord;
			// theta = 2*pi/neighbours
			// x = r*cos(i * theta) + r
			// y = r*sin(i * theta) + r
			// adding r for translating center to image center
			float x = (float)(radius)*cos(i*2.0*M_PI/neighbours) + (float)(radius);
			float y = (float)(radius)*sin(i*2.0*M_PI/neighbours) + (float)(radius);

			// relative indices
			neighbourhoodCoord[0] = floor(x); // fx
			neighbourhoodCoord[1] = floor(y); // fy
			neighbourhoodCoord[2] = ceil(x);  // cx
			neighbourhoodCoord[3] = ceil(y);  // cy

			// fractional parts
			float tx = x-floor(x);
			float ty = y-floor(y);

			// weights
			neighbourhoodCoord[4] = (1-tx) * (1-ty); // w1
			neighbourhoodCoord[5] =    tx  * (1-ty); // w2
			neighbourhoodCoord[6] = (1-tx) *    ty;  // w3
			neighbourhoodCoord[7] =    tx  *    ty;  // w4

			// add array to vector
			neighbourhoodCoords.push_back(neighbourhoodCoord);
		}
		return neighbourhoodCoords;
	}

	unsigned char minROR(unsigned char x, int numShifts){
		unsigned char m = x;
		for(int i=1; i<numShifts; i++){
			m = min((unsigned char)((x >> i)|(x << (CHAR_BIT-i))), m);
		}
		return m;
	}
}
