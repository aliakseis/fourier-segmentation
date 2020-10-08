/* This file implements the 2D Radix-2 Tree Sliding Window Discrete Fourier Transform algorithm 
described in the manuscript: "Algorithm xxx: The 2D Tree SWDFT Algorithm" */

#if 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <complex.h>
#include "tswdft2d.h"

#include <utility>
#include <vector>

const std::complex<double> I(0.0, 1.0);

/*  Gets the 1D node index based on the 2D binary tree position used by the tswdft2d function */
#define NODE(node0, node1) (((node0) * (n1)) + (node1))

/* Calculates the node index for a particular level of a tree */
#define IMOD(node, level) ((node) % (int) (pow((2), (level))))

/* Get index of tree inside the level_prev or level_cur arrays in the tswdft2d function */
#define TREE_2D(x, y) ((((x) * (N1)) + (y)) * ((n0) * (n1)))

/* Used to initialize level 0 of binary trees to the data  */
#define X_LOOKUP(x, y) (((x) * (N1)) + (y))

/* Used as a lookup for the output 4D array */
#define COEF_LOOKUP(x, y) ((((x) * (P1)) + (y)) * ((n0) * (n1)))

/* Lookups used in 2D SWFFT and 2D SWDFT sliding window functions  */
#define WINDOW_LOOKUP(x, y) (((x) * (n1)) + (y))

/* 
  2D Radix-2 Tree Sliding Window Discrete Fourier Transform 

  The tswdft2d function calculates the 2D discrete Fourier transform (DFT) for all n0 x n1 
  windows in the N0 x N1 array x. The function requires that n0 and n1 be powers of two. The 2D SWDFT 
  coefficients are output as a 4D array a with dimensions (N0 - n0 + 1) x (N1 - n1 + 1) x n0 x n1. 
  Both x and a are implemented as 1D arrays in row-major order. The algorithm corresponds to calculating 
  a two-dimensional tree underneath each element of x (other than the first few elements, which do not 
  have enough data). Each tree contains the calculations of the 2D Row-Column Fast Fourier Transform (FFT) 
  algorithm: which first takes 1D FFTs of each row, followed by 1D FFTs of the corresponding columns. 
  The algorithm first creates two twiddle factor vectors, which store the trigonometric constants used 
  for combining smaller DFTs during the FFT algorithm. Next, we allocate memory required for storing 2D FFT 
  calculations for each window position. The core of the algorithm is six nested loops: over row FFT levels, 
  over column FFT levels, over trees in the x-direction, trees in the y-direction, nodes in the 
  x-direction, and nodes in the y-direction. The calculation inside the loops is one complex 
  multiplication and one complex addition, where the complex multiplication involves the current tree and a 
  twiddle factor, and the complex addition involves a node from a shifted tree and the output of the 
  complex multiplication. 

  INPUT
  x  --- double complex * --- Row-major 2D array with dimensions N0 x N1
  n0 --- int              --- Window in the row-direction (must be power of two)
  n1 --- int              --- Window size in the column-direction (must be power of two)
  N0 --- int              --- Number of rows in x
  N1 --- int              --- Number of columns in x

  OUTPUT 
  a  --- double complex * --- Row-major 4D array with dimensions:
  			      (N0 - n0 + 1) x (N1 - n1 + 1) x n0 x n1
*/
std::complex<double> * tswdft2d(const std::complex<double> *x, int n0, int n1, int N0, int N1) {
	/* Initialize values integers for indexing, and complex-numbers for calculation inside the 6-loops */
	int shift0, shift1, level_min, level_max, nodes0, nodes1, min0, min1; 
    std::complex<double> T_prev, T_cur, twid;

	/* Verify n0 and n1 are powers of two */
	if ((int) ceil(log2(n0)) != (int) floor(log2(n0))) {
		printf("n0 is not a power of two! \n");
		exit(-1);
	}

	if ((int) ceil(log2(n1)) != (int) floor(log2(n1))) {
		printf("n1 is not a power of two! \n");
		exit(-1);
	}

	/* Calculate the number of levels in the binary tree */
	int m0 = (int) log2(n0);
	int m1 = (int) log2(n1);

	/* Initialize twiddle-factor vectors for both dimensions */ 
    //std::complex<double> *current_twiddle;
    std::vector<std::complex<double>> twiddle0(n0);
    std::vector<std::complex<double>> twiddle1(n1);

	for (int init_tf0 = 0; init_tf0 < n0; init_tf0++) {
		twiddle0[init_tf0] = pow(exp((2 * M_PI * I) / static_cast<double>(n0)), -init_tf0);
	}

	for (int init_tf1 = 0; init_tf1 < n1; init_tf1++) {
		twiddle1[init_tf1] = pow(exp((2 * M_PI * I) / static_cast<double>(n1)), -init_tf1);
	}

	/* Allocate memory for two levels of tree data-structure */ 
	auto level_prev = new std::complex<double>[N0 * N1 * n0 * n1];
	auto level_cur = new std::complex<double>[N0 * N1 * n0 * n1];
	
	/* Initialize level 0 of the tree data-structure to the data  */ 
	for (int row = 0; row < N0; row++) {
		for (int col = 0; col < N1; col++) {
			level_prev[TREE_2D(row, col)] = x[X_LOOKUP(row, col)];
		}
	}

	/* 
	  The six loops. The first two loops (dim and levels) are over levels of the tree data-structure. When dim = 1, 
	  the algorithm corresponds to the row FFT potion of the 2D FFT algorithm, and when dim = 0 the algorithm 
	  corresponds to the column FFT portion.The next two loops (N0 and N1) are over trees in both directions, 
	  and the final two loops (nodes0 and nodes1) are over nodes of a particular level of a particular tree. The 
	  calculation inside the six-loops uses macros TREE_2D, NODE, and IMOD to access the correct indices of 
	  the tree data-structure.
	*/ 
	for (int dim = 1; dim >= 0; dim--) {
	  /* Get levels of trees corresponding to this dimension */ 
	  level_min = (dim == 1) ? 1:(m1 + 1);
	  level_max = (dim == 1) ? m1:(m1 + m0);

	  /* Set the twiddle-factor vector corresponding to the current dimension */
	  auto& current_twiddle = (dim == 1) ? twiddle1:twiddle0; 

	  for (int level = level_min; level <= level_max; level++) {		
	    /* Number of nodes for this level of the binary tree in both direction*/			
		nodes0 = (dim == 1) ? 1:((int) pow(2, level - m1));
		nodes1 = (dim == 1) ? ((int) pow(2, level)):n1;

		/* Get shift distance between the current and previous tree with the repeated calculation */
		shift0 = (dim == 1) ? 0:((int) pow(2, m1 + m0 - level));
		shift1 = (dim == 1) ? ((int) pow(2, m1 - level)):0;

		/* Get first trees requiring this level for the row (min0) or column (min1) */  
		min0 = (dim == 1) ? 0:(n0 - shift0);
		min1 = (dim == 1) ? (n1 - shift1):(n1 - 1);

		for (int p0 = min0; p0 < N0; p0++) {
			for (int p1 = min1; p1 < N1; p1++) {
				for (int node0 = 0; node0 < nodes0; node0++) {
					for (int node1 = 0; node1 < nodes1; node1++) {
						T_prev = level_prev[TREE_2D(p0 - shift0, p1 - shift1) + IMOD(NODE(node0, node1), level - 1)];
						T_cur = level_prev[TREE_2D(p0, p1) + IMOD(NODE(node0, node1), level - 1)];
						twid = current_twiddle[(dim == 1) ? (node1 * shift1):(node0 * shift0)];

						level_cur[TREE_2D(p0, p1) + NODE(node0, node1)] = T_prev + (twid * T_cur);
					}
				}
			}
		}

		std::swap(level_prev, level_cur);
	  }
	}

	/* Subset the final DFT coefficients into the array a in row-major order */
	int result_ind = 0;
	for (int p0_result = (n0 - 1); p0_result < N0; p0_result++) {
		for (int p1_result = (n1 - 1); p1_result < N1; p1_result++) {
			memmove(&level_cur[result_ind], &level_prev[TREE_2D(p0_result, p1_result)], sizeof(std::complex<double>) * (n0 * n1));
			result_ind += (n0 * n1);
		}	
	}
	delete[] level_prev;

	//a = realloc(level_cur, sizeof(double complex) * (N0 - n0 + 1) * (N1 - n1 + 1) * n0 * n1);
	//return a;
    return level_cur;
}

std::complex<double> * tswdft2d(const unsigned char *x, int n0, int n1, int N0, int N1) {
    /* Initialize values integers for indexing, and complex-numbers for calculation inside the 6-loops */
    int shift0, shift1, level_min, level_max, nodes0, nodes1, min0, min1;
    std::complex<double> T_prev, T_cur, twid;

    /* Verify n0 and n1 are powers of two */
    if ((int)ceil(log2(n0)) != (int)floor(log2(n0))) {
        printf("n0 is not a power of two! \n");
        exit(-1);
    }

    if ((int)ceil(log2(n1)) != (int)floor(log2(n1))) {
        printf("n1 is not a power of two! \n");
        exit(-1);
    }

    /* Calculate the number of levels in the binary tree */
    int m0 = (int)log2(n0);
    int m1 = (int)log2(n1);

    /* Initialize twiddle-factor vectors for both dimensions */
    //std::complex<double> *current_twiddle;
    std::vector<std::complex<double>> twiddle0(n0);
    std::vector<std::complex<double>> twiddle1(n1);

    for (int init_tf0 = 0; init_tf0 < n0; init_tf0++) {
        twiddle0[init_tf0] = pow(exp((2 * M_PI * I) / static_cast<double>(n0)), -init_tf0);
    }

    for (int init_tf1 = 0; init_tf1 < n1; init_tf1++) {
        twiddle1[init_tf1] = pow(exp((2 * M_PI * I) / static_cast<double>(n1)), -init_tf1);
    }

    /* Allocate memory for two levels of tree data-structure */
    auto level_prev = new std::complex<double>[N0 * N1 * n0 * n1];
    auto level_cur = new std::complex<double>[N0 * N1 * n0 * n1];

    /* Initialize level 0 of the tree data-structure to the data  */
    for (int row = 0; row < N0; row++) {
        for (int col = 0; col < N1; col++) {
            level_prev[TREE_2D(row, col)] = double(x[X_LOOKUP(row, col)]) + 1.;
        }
    }

    /*
      The six loops. The first two loops (dim and levels) are over levels of the tree data-structure. When dim = 1,
      the algorithm corresponds to the row FFT potion of the 2D FFT algorithm, and when dim = 0 the algorithm
      corresponds to the column FFT portion.The next two loops (N0 and N1) are over trees in both directions,
      and the final two loops (nodes0 and nodes1) are over nodes of a particular level of a particular tree. The
      calculation inside the six-loops uses macros TREE_2D, NODE, and IMOD to access the correct indices of
      the tree data-structure.
    */
    for (int dim = 1; dim >= 0; dim--) {
        /* Get levels of trees corresponding to this dimension */
        level_min = (dim == 1) ? 1 : (m1 + 1);
        level_max = (dim == 1) ? m1 : (m1 + m0);

        /* Set the twiddle-factor vector corresponding to the current dimension */
        auto& current_twiddle = (dim == 1) ? twiddle1 : twiddle0;

        for (int level = level_min; level <= level_max; level++) {
            /* Number of nodes for this level of the binary tree in both direction*/
            nodes0 = (dim == 1) ? 1 : ((int)pow(2, level - m1));
            nodes1 = (dim == 1) ? ((int)pow(2, level)) : n1;

            /* Get shift distance between the current and previous tree with the repeated calculation */
            shift0 = (dim == 1) ? 0 : ((int)pow(2, m1 + m0 - level));
            shift1 = (dim == 1) ? ((int)pow(2, m1 - level)) : 0;

            /* Get first trees requiring this level for the row (min0) or column (min1) */
            min0 = (dim == 1) ? 0 : (n0 - shift0);
            min1 = (dim == 1) ? (n1 - shift1) : (n1 - 1);

            for (int p0 = min0; p0 < N0; p0++) {
                for (int p1 = min1; p1 < N1; p1++) {
                    for (int node0 = 0; node0 < nodes0; node0++) {
                        for (int node1 = 0; node1 < nodes1; node1++) {
                            T_prev = level_prev[TREE_2D(p0 - shift0, p1 - shift1) + IMOD(NODE(node0, node1), level - 1)];
                            T_cur = level_prev[TREE_2D(p0, p1) + IMOD(NODE(node0, node1), level - 1)];
                            twid = current_twiddle[(dim == 1) ? (node1 * shift1) : (node0 * shift0)];

                            level_cur[TREE_2D(p0, p1) + NODE(node0, node1)] = T_prev + (twid * T_cur);
                        }
                    }
                }
            }

            std::swap(level_prev, level_cur);
        }
    }

    /* Subset the final DFT coefficients into the array a in row-major order */
    int result_ind = 0;
    for (int p0_result = (n0 - 1); p0_result < N0; p0_result++) {
        for (int p1_result = (n1 - 1); p1_result < N1; p1_result++) {
            memmove(&level_cur[result_ind], &level_prev[TREE_2D(p0_result, p1_result)], sizeof(std::complex<double>) * (n0 * n1));
            result_ind += (n0 * n1);
        }
    }
    delete[] level_prev;

    //a = realloc(level_cur, sizeof(double complex) * (N0 - n0 + 1) * (N1 - n1 + 1) * n0 * n1);
    //return a;
    return level_cur;
}

#endif