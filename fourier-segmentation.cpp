// fourier-segmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "tswdft2d.h"

#include "tsne.h"
#include "splittree.h"
#include "vptree.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/ml.hpp"

#include <dlib/dnn.h>
#include <dlib/svm_threaded.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

static const auto M_PI = 3.14159265358979323846;

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

//DEFAULT_NO_DIMS = 2
//INITIAL_DIMENSIONS = 50
//DEFAULT_PERPLEXITY = 50
//DEFAULT_THETA = 0.5
//EMPTY_SEED = -1
//DEFAULT_USE_PCA = True
//DEFAULT_MAX_ITERATIONS = 1000

// Function that runs the Barnes-Hut implementation of t-SNE
double* tSNE(double* X, int N, int D, //double* Y,
    int no_dims = 2, double perplexity = 30, double theta = .5,
    int num_threads = 10, int max_iter = 1000, int n_iter_early_exag = 250,
    int random_state = -1, bool init_from_Y = false, int verbose = 0,
    double early_exaggeration = 12, double learning_rate = 200,
    double *final_error = NULL, int distance = 1) {

    //// Define some variables
    //int origN, N, D, no_dims, max_iter;
    //double perplexity, theta, *data;
    //int rand_seed = -1;

    //// Read the parameters and the dataset
    //if (TSNE::load_data(&data, &origN, &D, &no_dims, &theta, &perplexity, &rand_seed, &max_iter)) {

        // Make dummy landmarks
        //int N = origN;
        //int* landmarks = (int*)malloc(N * sizeof(int));
        //if (landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        //for (int n = 0; n < N; n++) landmarks[n] = n;

        // Now fire up the SNE implementation
        auto Y = new double[N * no_dims * sizeof(double)]();

        //double* costs = (double*)calloc(N, sizeof(double));
        //if (Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        //TSNE::run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter, 250, 250);

        TSNE<SplitTree, euclidean_distance_squared> tsne;
        tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, n_iter_early_exag,
            random_state, init_from_Y, verbose, early_exaggeration, learning_rate, final_error);


        // Save the results
        //TSNE::save_data(Y, landmarks, costs, N, no_dims);

        // Clean up the memory
        //free(data); data = NULL;
        //free(Y); Y = NULL;
        //free(costs); costs = NULL;
        //free(landmarks); landmarks = NULL;

        return Y;
    //}
}

//////////////////////////////////////////////////////////////////////////////

void generatePolyMembers(
    const std::vector<double>& input,
    int power, int idx, double product, std::vector<double>& result)
{
    if (power == 0)
    {
        if (idx != -1)
            result.push_back(product);
        return;
    }

    if (idx == -1)
    {
        generatePolyMembers(input, power - 1, idx, product, result);
        ++idx;
    }

    for (; idx < input.size(); ++idx)
    {
        generatePolyMembers(input, power - 1, idx, product * input[idx], result);
    }
}

std::vector<double> generatePolyMembers(const std::vector<double>& input, int power)
{
    std::vector<double> result;
    generatePolyMembers(input, power, -1, 1., result);
    return result;
}

//int sign(double v)
//{
//    return (v > 0) ? 1 : ((v < 0) ? -1 : 0);
//}

auto linearRegression(const cv::Mat& src, const std::vector<double>& prices)
{
    enum { MAX_POLY_POWER = 3 };

    //int numFeatures, numRows;
    //std::cin >> numFeatures >> numRows;

    const int numRows = src.rows;
    const int numFeatures = src.cols;

    std::vector<std::vector<double>> data;
    //std::vector<double> prices;
    for (int i = 0; i < numRows; ++i)
    {
        std::vector<double> row(numFeatures);
        for (int j = 0; j < numFeatures; ++j)
        {
            //double v;
            //std::cin >> v;
            //row.push_back(v);
            row[j] = src.at<double>(i, j);
        }
        data.push_back(generatePolyMembers(row, MAX_POLY_POWER));

        //double v;
        //std::cin >> v;
        //prices.push_back(v);
    }

    const auto num_params = data[0].size();

    std::vector<double> avg(num_params, 0.);
    std::vector<double> dev(num_params, 0.);

    for (const auto& l : data)
    {
        for (int i = 0; i < num_params; ++i)
        {
            avg[i] += l[i];
            dev[i] += l[i] * l[i];
        }
    }

    for (int i = 0; i < num_params; ++i)
    {
        avg[i] /= numRows;
        dev[i] = sqrt(dev[i] / numRows - avg[i] * avg[i]);
    }

    for (auto& l : data)
    {
        for (int i = 0; i < num_params; ++i)
        {
            l[i] = (l[i] - avg[i]) / dev[i];
        }
    }

    for (auto& l : data)
    {
        l.insert(l.begin(), 1.);
    }

    std::vector<double> w(num_params + 1, -1.);

    enum { N_ITER = 100000 };
    
    //const double lambda = 1.;
    const double lambda = .5;
    const double lr = 0.1;

    double prev_sq_dist = DBL_MAX;

    for (int i = 0; i < N_ITER; ++i)
    {
        double sq_dist = 0.;

        std::vector<double> delta_l(num_params + 1, 0.);
        for (int i = 0; i < numRows; ++i)
        {
            const auto& l = data[i];
            const auto delta = std::inner_product(l.begin(), l.end(), w.begin(), 0.) - prices[i];
            sq_dist += delta * delta;
            for (int j = 0; j < delta_l.size(); ++j)
                delta_l[j] += l[j] * delta;
        }

        if (sq_dist >= prev_sq_dist)
            break;
        prev_sq_dist = sq_dist;

        for (int i = 0; i < delta_l.size(); ++i)
        {
            const auto delta = delta_l[i] / numRows + lambda * sign(w[i]);
            w[i] -= lr * delta;
        }
    }

    //int num_tests;
    //std::cin >> num_tests;

    std::vector<double> results;

    for (int i = 0; i < numRows; ++i)
    {
        std::vector<double> row(numFeatures);
        for (int j = 0; j < numFeatures; ++j)
        {
            //double v;
            //std::cin >> v;
            //row.push_back(v);
            row[j] = src.at<double>(i, j);
        }
        //data.push_back(generatePolyMembers(row, MAX_POLY_POWER));

        auto l = generatePolyMembers(row, MAX_POLY_POWER);
        for (int i = 0; i < num_params; ++i)
        {
            l[i] = (l[i] - avg[i]) / dev[i];
        }
        l.insert(l.begin(), 1.);

        const auto result = std::inner_product(l.begin(), l.end(), w.begin(), 0.);

        results.push_back(result);

        //double v;
        //std::cin >> v;
        //prices.push_back(v);
    }

    return results;
}

double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

auto logisticRegression(const cv::Mat& src, const std::vector<unsigned long>& prices, const cv::Mat& test)
{
    enum { MAX_POLY_POWER = 3 };

    //int numFeatures, numRows;
    //std::cin >> numFeatures >> numRows;

    const int numRows = src.rows;
    const int numFeatures = src.cols;

    std::vector<std::vector<double>> data;
    //std::vector<double> prices;
    for (int i = 0; i < numRows; ++i)
    {
        std::vector<double> row(numFeatures);
        for (int j = 0; j < numFeatures; ++j)
        {
            //double v;
            //std::cin >> v;
            //row.push_back(v);
            row[j] = src.at<float>(i, j);
        }
        data.push_back(generatePolyMembers(row, MAX_POLY_POWER));

        //double v;
        //std::cin >> v;
        //prices.push_back(v);
    }

    const auto num_params = data[0].size();

    std::vector<double> avg(num_params, 0.);
    std::vector<double> dev(num_params, 0.);

    for (const auto& l : data)
    {
        for (int i = 0; i < num_params; ++i)
        {
            avg[i] += l[i];
            dev[i] += l[i] * l[i];
        }
    }

    for (int i = 0; i < num_params; ++i)
    {
        avg[i] /= numRows;
        dev[i] = sqrt(dev[i] / numRows - avg[i] * avg[i]);
    }

    for (auto& l : data)
    {
        for (int i = 0; i < num_params; ++i)
        {
            l[i] = (l[i] - avg[i]) / dev[i];
        }
    }

    for (auto& l : data)
    {
        l.insert(l.begin(), 1.);
    }

    std::vector<double> w(num_params + 1, -1.);

    enum { N_ITER = 100000 };

    //const double lambda = 1.;
    //const double lambda = .5;
    //const double lr = 0.1;

    const double lambda = .015;
    const double lr = .97;

    double prev_sq_dist = DBL_MAX;

    for (int i = 0; i < N_ITER; ++i)
    {
        double sq_dist = 0.;

        std::vector<double> delta_l(num_params + 1, 0.);
        for (int i = 0; i < numRows; ++i)
        {
            const auto& l = data[i];
            const auto delta = sigmoid(std::inner_product(l.begin(), l.end(), w.begin(), *w.rbegin())) - prices[i];
            sq_dist += delta * delta;
            //for (int j = 0; j < delta_l.size(); ++j)
            //    delta_l[j] += l[j] * delta;
            for (int j = 0; j < delta_l.size() - 1; ++j)
                delta_l[j] += l[j] * delta;

            delta_l[delta_l.size() - 1] += delta;
        }

        if (sq_dist >= prev_sq_dist)
            break;
        prev_sq_dist = sq_dist;

        for (int i = 0; i < delta_l.size(); ++i)
        {
            const auto delta = delta_l[i] / numRows + lambda * sign(w[i]);
            w[i] -= lr * delta;
        }
    }

    //int num_tests;
    //std::cin >> num_tests;

    const int num_tests = test.rows;

    std::vector<unsigned long> results;

    for (int i = 0; i < num_tests; ++i)
    {
        std::vector<double> row(numFeatures);
        for (int j = 0; j < numFeatures; ++j)
        {
            //double v;
            //std::cin >> v;
            //row.push_back(v);
            row[j] = test.at<float>(i, j);
        }
        //data.push_back(generatePolyMembers(row, MAX_POLY_POWER));

        auto l = generatePolyMembers(row, MAX_POLY_POWER);
        for (int i = 0; i < num_params; ++i)
        {
            l[i] = (l[i] - avg[i]) / dev[i];
        }
        l.insert(l.begin(), 1.);

        const auto result = sigmoid(std::inner_product(l.begin(), l.end(), w.begin(), 0.)) > 0.5;

        results.push_back(result);

        //double v;
        //std::cin >> v;
        //prices.push_back(v);
    }

    return results;
}


//////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    /*Read Image*/
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    enum { IMG_DIMENSION = 720 };
    resize(img, img, cv::Size(IMG_DIMENSION, IMG_DIMENSION), 0, 0, cv::INTER_LANCZOS4);


    //const auto kernel_size = 3;
    //cv::GaussianBlur(img, img, cv::Size(kernel_size, kernel_size), 0, 0, cv::BORDER_DEFAULT);


    enum { WINDOW_DIMENSION = 16 };
    //volatile 
    auto transformed = tswdft2d<double>(img.data, WINDOW_DIMENSION, WINDOW_DIMENSION, img.rows, img.cols);

    const auto numValues = (img.rows - WINDOW_DIMENSION + 1) * (img.cols - WINDOW_DIMENSION + 1);

/*
    cv::Mat pcaInput(numValues, WINDOW_DIMENSION * WINDOW_DIMENSION - 1, CV_64FC1);

    for (int i = 0; i < numValues; ++i)
        for (int j = 1; j < WINDOW_DIMENSION * WINDOW_DIMENSION; ++j)
            //pcaInput.at<double>(i, j - 1) = transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j].real();
            //pcaInput.at<double>(i, j - 1) = std::abs(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
            pcaInput.at<double>(i, j - 1) = std::arg(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
*/

    cv::Mat pcaInput(numValues, WINDOW_DIMENSION * WINDOW_DIMENSION * 2 - 1, CV_64FC1);

    for (int i = 0; i < numValues; ++i)
    {
        //if (i == 0)//numValues - 1)
        //{
        //    for (int j = 0; j < WINDOW_DIMENSION * WINDOW_DIMENSION; ++j)
        //        std::cout << transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j] << ' ';

        //    std::cout << '\n';
        //}

        pcaInput.at<double>(i, 0) = transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION].real();

        for (int j = 1; j < WINDOW_DIMENSION * WINDOW_DIMENSION; ++j)
        {
            //pcaInput.at<double>(i, j - 1) = transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j].real();
            //pcaInput.at<double>(i, j - 1) = std::abs(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
            //pcaInput.at<double>(i, j - 1) = std::arg(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);

            //pcaInput.at<double>(i, j * 2 - 1) = transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j].real();
            //pcaInput.at<double>(i, j * 2) = transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j].imag();

            //pcaInput.at<double>(i, j * 2 - 1) = std::abs(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
            //pcaInput.at<double>(i, j * 2) = std::arg(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
            const auto amplitude = std::abs(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
            pcaInput.at<double>(i, j * 2 - 1) = amplitude;
            pcaInput.at<double>(i, j * 2) = amplitude * std::arg(transformed[i * WINDOW_DIMENSION * WINDOW_DIMENSION + j]) * 2. / M_PI;
        }
    }

    const auto visualizationRows = img.rows - WINDOW_DIMENSION + 1;
    const auto visualizationCols = img.cols - WINDOW_DIMENSION + 1;

    cv::Mat sizes;

    for (int i = 0; i < numValues; ++i)
    {
        auto r = pcaInput.row(i);
        cv::normalize(r, r);

        //double addition[2]{ double(i / visualizationCols) / visualizationRows / 8., double(i % visualizationCols) / visualizationCols / 8. };
        //sizes.push_back(cv::Mat(1, 2, CV_64FC1, addition));

        //if (i == numValues - 1)
        //    std::cout << r << '\n';
    }

    //cv::hconcat(pcaInput, sizes, pcaInput);

    //delete[] transformed;
    //transformed = nullptr;
    
    decltype(transformed)().swap(transformed);
    //transformed.clear();

    /*
    volatile auto tsne = tSNE((double*)pcaInput.data, numValues, WINDOW_DIMENSION * WINDOW_DIMENSION - 1);

    double maxrad = 0;
    for (int i = 0; i < numValues; ++i)
    {
        maxrad = std::max(maxrad, hypot(tsne[i * 2], tsne[i * 2 + 1]));
    }

    const auto visualizationRows = img.rows - WINDOW_DIMENSION + 1;
    const auto visualizationCols = img.cols - WINDOW_DIMENSION + 1;
    cv::Mat visualization(visualizationRows, visualizationCols, CV_8UC3);

    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;
            visualization.at<cv::Vec3b>(y, x)
                = computeColor(tsne[sourceOffset * 2] / maxrad, tsne[sourceOffset * 2 + 1] / maxrad);
        }

    cv::imshow("visualization", visualization);
    */

    //*
    cv::PCA pca(pcaInput,         //Input Array Data
        cv::Mat(),                //Mean of input array, if you don't want to pass it   simply put Mat()
        cv::PCA::DATA_AS_ROW,     //int flag
        15);                       // number of component that you want to retain(keep)

    auto reduced = pca.project(pcaInput);

    double maxrad = 0;
    for (int i = 0; i < numValues; ++i)
    {
        maxrad = std::max(maxrad, hypot(reduced.at<double>(i, 0), reduced.at<double>(i, 1)));
    }

    cv::Mat visualization(visualizationRows, visualizationCols, CV_8UC3);

    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;
            visualization.at<cv::Vec3b>(y, x) 
                = computeColor(reduced.at<double>(sourceOffset, 0) / maxrad, reduced.at<double>(sourceOffset, 1) / maxrad);
        }

    cv::imshow("visualization", visualization);
    //*/


    /*
    std::vector<double> prices;
    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            prices.push_back(img.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2));
        }

    auto regressed = linearRegression(reduced, prices);
    cv::Mat rawRegressed(cv::Size(visualizationCols, visualizationRows), CV_64FC1, regressed.data());

    cv::Mat imgRegressed;
    rawRegressed.convertTo(imgRegressed, CV_8U);

    cv::imshow("regressed", imgRegressed);
    //*/

    //*
    cv::Mat threshold;
    cv::adaptiveThreshold(img, threshold, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 7, 2.);

    cv::imshow("threshold", threshold);

    cv::Mat imgNeg = 255 - img;
    cv::Mat thresholdNeg;
    cv::adaptiveThreshold(imgNeg, thresholdNeg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 7, 2.);

    cv::imshow("thresholdNeg", thresholdNeg);

    //cv::waitKey(0);


    //std::vector<float> labels;

    cv::Mat reducedFloat;
    reduced.convertTo(reducedFloat, CV_32F);

/*
    int counters[2]{};

    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            //labels.push_back((threshold.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2) > 0) ? 1.f : -1.f);
            const bool raised = (threshold.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2) > 0);
            ++counters[raised];
        }

    std::cout << counters[0] << " : " << counters[1] << '\n';

    //cv::Mat labels(numValues, 1, CV_32F, -1.f);

    cv::Mat labels, training;

    int idx = 0;
    int exLeft = counters[0];
    int exNeeded = counters[1];

    std::default_random_engine dre;

    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            bool toAdd = true;
            //labels.push_back((threshold.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2) > 0) ? 1.f : -1.f);
            const bool raised = (threshold.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2) > 0);

            if (!raised && exLeft)
            {
                --exLeft;
                std::uniform_int_distribution<int> di(0, exLeft);
                const int outcome = di(dre);
                if (outcome < exNeeded)
                {
                    --exNeeded;
                }
                else
                {
                    toAdd = false;
                }
            }

            //labels.at<float>(idx, 0) = raised? 1.f : -1.f;
            if (toAdd)
            {
                training.push_back(reducedFloat.row(idx));
                labels.push_back(raised ? 1.f : -1.f);
            }

            ++idx;
        }
*/

    cv::Mat labels, training;

    std::vector<unsigned long> ullabels;

    int counters[2]{};
    int idx = 0;
    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            bool toAdd = true;
            //labels.push_back((threshold.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2) > 0) ? 1.f : -1.f);
            const bool raised = (threshold.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2) > 0);
            const bool sunken = (thresholdNeg.at<uchar>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2) > 0);

            if (raised && !sunken)
            {
                training.push_back(reducedFloat.row(idx));
                labels.push_back(1.f);
                ullabels.push_back(1);
                ++counters[1];
            }
            else if (sunken)
            {
                training.push_back(reducedFloat.row(idx));
                labels.push_back(-1.f);
                ullabels.push_back(0);
                ++counters[0];
            }

            ++idx;
        }

    std::cout << counters[0] << " : " << counters[1] << '\n';


    /*
    auto nn = cv::ml::ANN_MLP::create();

    cv::Mat_<int> layers(4, 1);
    layers << 10, 20, 10, 1;

    //cv::Mat_<int> layers(3, 1);
    //layers << 4, 10, 1;

    nn->setLayerSizes(layers);
    nn->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);
    nn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 2.35, 1.0);
    nn->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300, 0.00001));

    nn->train(training, 0, labels); // yes, that'll take a few minutes ..

    //std::vector<float> result;
    cv::Mat result;
    nn->predict(reducedFloat, result);
    */



    /*
    auto lr1 = cv::ml::LogisticRegression::create();
    lr1->setLearningRate(0.001);
    lr1->setIterations(1000);
    lr1->setRegularization(cv::ml::LogisticRegression::REG_L2);
    lr1->setTrainMethod(cv::ml::LogisticRegression::BATCH);
    lr1->setMiniBatchSize(10);
    //! [init]
    lr1->train(training, cv::ml::ROW_SAMPLE, labels);

    cv::Mat result;
    lr1->predict(reducedFloat, result);
    //*/

    /*
    auto boost = cv::ml::Boost::create();
    boost->setBoostType(cv::ml::Boost::DISCRETE);
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(2);
    boost->setUseSurrogates(false);
    boost->setPriors(cv::Mat());

    boost->train(training, cv::ml::ROW_SAMPLE, labels);

    cv::Mat result;
    boost->predict(reducedFloat, result);
    */

    /*
    auto rt = cv::ml::RTrees::create();

    rt->train(training, cv::ml::ROW_SAMPLE, labels);

    cv::Mat result;
    rt->predict(reducedFloat, result);
    */

    /*
    using DType = float;
    using Matrix = dlib::matrix<DType>;

    using namespace dlib;

    std::vector<Matrix> samples;
    for (int i = 0; i < training.rows; ++i)
    {
        auto r = training.row(i);
        samples.push_back(dlib::mat((float*)r.data, r.cols));
    }

    std::vector<Matrix> test_data;

    for (int i = 0; i < reducedFloat.rows; ++i)
    {
        auto r = reducedFloat.row(i);
        test_data.push_back(dlib::mat((float*)r.data, r.cols));
    }
    //*/

#if 0
    /*
    using net_type = loss_multiclass_log<
        //fc<3, relu<fc<10, relu<fc<5, input<matrix<DType>>>>>>>>;
        //fc<2, relu<fc<10, relu<fc<20, input<matrix<DType>>>>>>>>;
        fc<2, relu<fc<20, input<matrix<DType>>>>>>;
    //fc<2, relu<fc<10, relu<fc<20, relu<fc<10, input<matrix<DType>>>>>>>>>>;
    */

    using net_type = loss_multiclass_log<
        //fc<2, htan<fc<8, htan<fc<16, htan<fc<32, input<matrix<DType>>>>>>>>>>;
        fc<2, htan<fc<10, htan<fc<20, input<matrix<DType>>>>>>>>;
        //fc<2, htan<fc<20, input<matrix<DType>>>>>>;

    net_type net;

    //float weight_decay = 0.0001f;
    float weight_decay = 0.003f;
    float momentum = 0.5f;
    sgd solver(weight_decay, momentum);

    dnn_trainer<net_type> trainer(net, solver);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(8);
    trainer.be_verbose();

    trainer.set_max_num_epochs(1000);



    trainer.train(samples, ullabels);
    net.clean(); // Clean some intermediate data which is not used for evaluation




    auto predicted_labels = net(test_data);
#endif

/*
    typedef Matrix SampleType;

    using OVOtrainer = one_vs_one_trainer<any_trainer<SampleType>>;
    using KernelType = radial_basis_kernel<SampleType>;

    krr_trainer<KernelType> krr_trainer;
    krr_trainer.set_kernel(KernelType(0.1));

    OVOtrainer trainer;
    trainer.set_trainer(krr_trainer);

    one_vs_one_decision_function<OVOtrainer> df = trainer.train(samples, labels);

    //Classes classes;
    //DataType accuracy = 0;
    std::vector<size_t> predicted_labels;
    for (size_t i = 0; i != test_data.size(); i++) {
        auto vec = test_data[i];
        auto class_idx = static_cast<size_t>(df(vec));
        predicted_labels.push_back(class_idx);
    }
*/

/*
    auto nn = cv::ml::KNearest::create(); //cv::ml::ANN_MLP::create();

    nn->setDefaultK(51);

    nn->train(training, 0, labels); // yes, that'll take a few minutes ..

    cv::Mat result;
    nn->predict(reducedFloat, result);
*/

    auto predicted_labels = logisticRegression(training, ullabels, reducedFloat);

    cv::Mat trained(visualizationRows, visualizationCols, CV_8UC1);
    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;
            //trained.at<uchar>(y, x) = (result.at<float>(sourceOffset, 0) > 0.55) ? 255 : 0;
            trained.at<uchar>(y, x) = predicted_labels[sourceOffset] ? 255 : 0;
        }

    cv::imshow("trained", trained);
    //*/



    cv::waitKey(0);

    if (argc > 2)
    {
        cv::imwrite(argv[2], trained);
    }

    return 0;
}

