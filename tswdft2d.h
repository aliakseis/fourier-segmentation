// https://github.com/leerichardson/tree-swdft-2D

#pragma once

#include <math.h>
#include <complex>

#define M_PI 3.14159265358979323846
//#define max(a, b) (((a) > (b)) ? (a) : (b)) 


/* Declare all C programs in this package  */

/* The 2D Radix-2 Tree Sliding Window Discrete Fourier Transform */ 
//void swap(double complex **a, double complex **b);
std::complex<double> * tswdft2d(const std::complex<double> *x, int n0, int n1, int N0, int N1);

std::complex<double> * tswdft2d(const unsigned char *x, int n0, int n1, int N0, int N1);

///* The 2D Sliding Window discrete Fourier transform */ 
//std::complex<double> * swdft2d(std::complex<double> *x, int n0, int n1, int N0, int N1);
//
///* The 2D Sliding Window Fast Fourier Transform */
//std::complex<double> * swfft2d(std::complex<double> *x, int n0, int n1, int N0, int N1);
