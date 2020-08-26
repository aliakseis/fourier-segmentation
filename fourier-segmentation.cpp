// fourier-segmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "tswdft2d.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <algorithm>
#include <iostream>

// https://gist.github.com/voidqk/fc5a58b7d9fc020ecf7f2f5fc907dfa5
inline float fastAtan2_(float y, float x)
{
    static const float c1 = M_PI / 4.0;
    static const float c2 = M_PI * 3.0 / 4.0;
    //if (y == 0 && x == 0)
    //    return 0;

    if (y == 0)
        return 0;
    if (x == 0)
        return (y > 0) ? (M_PI / 2.) : (M_PI / 2.);

    float abs_y = fabsf(y);
    float angle;
    if (x >= 0)
        angle = c1 - c1 * ((x - abs_y) / (x + abs_y));
    else
        angle = c2 - c1 * ((x + abs_y) / (abs_y - x));
    if (y < 0)
        return -angle;
    return angle;
}


static cv::Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static cv::Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    //const float rad = hypot(fx, fy); //sqrt(fx * fx + fy * fy);
    const float rad = std::max(std::abs(fx), std::abs(fy)); //sqrt(fx * fx + fy * fy);
    const float a = fastAtan2_(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const auto col0 = colorWheel[k0][b];// / 255.f;
        const auto col1 = colorWheel[k1][b];// / 255.f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 255 - rad * (255 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(/*255.f * */col);
    }

    return pix;
}


int main(int argc, char *argv[])
{
    /*Read Image*/
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    enum { IMG_DIMENSION = 512 };
    resize(img, img, cv::Size(IMG_DIMENSION, IMG_DIMENSION), 0, 0, cv::INTER_LANCZOS4);

    enum { WINDOW_DIMENSION = 32 };
    //volatile 
    auto transformed = tswdft2d(img.data, WINDOW_DIMENSION, WINDOW_DIMENSION, img.rows, img.cols);

    const auto numValues = (img.rows - WINDOW_DIMENSION + 1) * (img.cols - WINDOW_DIMENSION + 1);

    cv::Mat pcaInput(numValues, WINDOW_DIMENSION * WINDOW_DIMENSION - 1, CV_32FC1);

    for (int i = 0; i < numValues; ++i)
        for (int j = 1; j < WINDOW_DIMENSION * WINDOW_DIMENSION; ++j)
            pcaInput.at<float>(i, j - 1) = transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j].real();

    for (int i = 0; i < numValues; ++i)
    {
        auto r = pcaInput.row(i);
        cv::normalize(r, r);
    }

    delete[] transformed;
    transformed = nullptr;

    cv::PCA pca(pcaInput,         //Input Array Data
        cv::Mat(),                //Mean of input array, if you don't want to pass it   simply put Mat()
        cv::PCA::DATA_AS_ROW,     //int flag
        2);                       // number of component that you want to retain(keep)

    auto reduced = pca.project(pcaInput);

    float maxrad = 0;
    for (int i = 0; i < numValues; ++i)
    {
        maxrad = std::max(maxrad, hypot(reduced.at<float>(i, 0), reduced.at<float>(i, 1)));
    }

    const auto visualizationRows = img.rows - WINDOW_DIMENSION + 1;
    const auto visualizationCols = img.cols - WINDOW_DIMENSION + 1;
    cv::Mat visualization(visualizationRows, visualizationCols, CV_8UC3);

    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;
            visualization.at<cv::Vec3b>(y, x) 
                = computeColor(reduced.at<float>(sourceOffset, 0) / maxrad, reduced.at<float>(sourceOffset, 1) / maxrad);
        }

    cv::imshow("visualization", visualization);

    cv::waitKey(0);

    if (argc > 2)
    {
        cv::imwrite(argv[2], visualization);
    }

    return 0;
}

