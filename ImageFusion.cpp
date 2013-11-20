//
//  ImageFusion.cpp
//  Fuse
//
//  Created by Swechha Prakash on 10/11/13.
//  Copyright (c) 2013 Swechha Prakash. All rights reserved.
//

#include "ImageFusion.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

double E=2.718281828459;
int LEVELS=3;
int wC=1, wS=1, wE=1;

using namespace cv;
using namespace std;

void imageFusion(Mat, Mat, int, int, int);
Mat getContrastMatrix(Mat);
Mat getSaturationMatrix(Mat, Mat);
Mat getExposureMatrix(Mat, double);
Mat getWeightMatrix(Mat, Mat, Mat);
vector<Mat> getGuassianPyr(Mat);
vector<Mat> getLaplacianPyr(Mat);
vector<Mat> computeBlenPyr(vector<Mat>, vector<Mat>, vector<Mat>, vector<Mat>);
Mat collapseLaplacianPyr(vector<Mat>);
Mat getNormalisedMatrix(Mat, Mat);



int init(Mat img1, Mat img2)
{
    
//    Mat img1 = imread("sample.jpg", CV_LOAD_IMAGE_COLOR);
//    Mat img2 = imread("sample2.jpg", CV_LOAD_IMAGE_COLOR);
    
    if (img1.empty() || img2.empty()) {
        cout<<"Could not load the image"<<endl;
        return -1;
    }
    
    cout<<"Image loaded"<<endl;
    
    imageFusion(img1, img2, 1, 1, 1);
    
    return 0;
}


/*
 * source1, source2: Source Images
 * contrast, saturation, exposure: weightage to each of the parameters
 * Final image after fusion is written to "fusedImage.jpg"
 */
void imageFusion(Mat source1, Mat source2, int contrast, int saturation, int exposure)
{
    int rows = source1.rows;
    int columns = source1.cols;
    
    wC = contrast;
    wS = saturation;
    wE = exposure;
    
    Mat gray1 = cvCreateMat(rows, columns, CV_8UC1);
    Mat gray2 = cvCreateMat(rows, columns, CV_8UC1);
    
    //grayscale intensities for saturation calculation
    cvtColor(source1, gray1, CV_BGR2GRAY);
    cvtColor(source2, gray2, CV_BGR2GRAY);
    
    if (gray1.empty() || gray2.empty()) {
        cout<<"Conversion to grayscale failed"<<endl;
        exit(-1);
    }
    
    
    
    Mat contrastMatrix1 = getContrastMatrix(gray1);
    contrastMatrix1.convertTo(contrastMatrix1, CV_32F, 1.0/255.0);
    
    source1.convertTo(source1, CV_32FC3, 1.0/255.0);
    gray1.convertTo(gray1, CV_32FC1, 1.0/255.0);
    Mat saturationMatirx1 = getSaturationMatrix(source1, gray1);
    Mat exposureMatrix1 = getExposureMatrix(source1, 0.2);
    
    Mat contrastMatrix2 = getContrastMatrix(gray2);
    contrastMatrix2.convertTo(contrastMatrix2, CV_32F, 1.0/255.0);
    
    source2.convertTo(source2, CV_32FC3, 1.0/255.0);
    gray2.convertTo(gray2, CV_32FC1, 1.0/255.0);
    Mat saturationMatirx2 = getSaturationMatrix(source2, gray2);
    Mat exposureMatrix2 = getExposureMatrix(source2, 0.2);
    
    
    
    
    Mat weight1 = getWeightMatrix(contrastMatrix1, saturationMatirx1, exposureMatrix1);
    Mat weight2 = getWeightMatrix(contrastMatrix2, saturationMatirx2, exposureMatrix2);
    
    Mat temp = getNormalisedMatrix(weight1, weight2);
    weight2 = getNormalisedMatrix(weight2, weight1);
    weight1 = temp;
    
    
    vector<Mat> gPyr1 = getGuassianPyr(weight1);
    vector<Mat> lPyr1 = getLaplacianPyr(source1);
    vector<Mat> gPyr2 = getGuassianPyr(weight2);
    vector<Mat> lPyr2 = getLaplacianPyr(source2);
    
    vector<Mat> blended = computeBlenPyr(gPyr1, gPyr2, lPyr1, lPyr2);
    
    
    Mat result = collapseLaplacianPyr(blended);
    imwrite("final.jpg", result);
}

/*
 * Generating contrast matrix
 * absolute of the laplacian of input matrix
 */
Mat getContrastMatrix(Mat input)
{
    Mat contrast = cvCreateMat(input.rows, input.cols, CV_32FC3);
    Mat laplacian = cvCreateMat(input.rows, input.cols, CV_8UC3);
    
    Laplacian(input, laplacian, 0);
    contrast = abs(laplacian);
    
    //imwrite("laplacian.jpg", contrast);
    
    return contrast;
}

/*
 * Generating saturation matrix
 * standard deviation within RGB
 */
Mat getSaturationMatrix(Mat input, Mat grayscale)
{
    Mat saturation = cvCreateMat(input.rows, input.cols, CV_32FC1);
    
    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            float r = input.at<Vec3f>(i,j)[0];
            float g = input.at<Vec3f>(i,j)[1];
            float b = input.at<Vec3f>(i,j)[2];
            float u = grayscale.at<float>(i,j);
            
            float sat = (sqrt(pow(double(r-u), 2.0) + pow(double(g-u), 2.0) + pow(double(b-u), 2.0)))/3.0;
            
            saturation.at<float>(i,j) = sat;
        }
    }
    cout<<"Finished writing saturation matrix"<<endl;
    
    return saturation;
}


/*
 * Generating exposure matrix
 * exp{-(i-0.5)^2/2(sigma^2)}; i=intensity
 * calculated across 3 channels and product is returned finally
 */
Mat getExposureMatrix(Mat input, double sigma)
{
    Mat exposure = cvCreateMat(input.rows, input.cols, CV_32FC1);
    //input.convertTo(input, CV_32FC3);
    
    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            float r = input.at<Vec3f>(i,j)[0];
            float g = input.at<Vec3f>(i,j)[1];
            float b = input.at<Vec3f>(i,j)[2];
            
            float er = pow(E, (-((r-0.5)*(r-0.5))/(2*sigma*sigma)));
            float eg = pow(E, (-((g-0.5)*(g-0.5))/(2*sigma*sigma)));
            float eb = pow(E, (-((b-0.5)*(b-0.5))/(2*sigma*sigma)));
            
            float exps = (er*eg*eb);
            
            exposure.at<float>(i,j) = exps;
        }
    }
    cout<<"Finished writing exposure matrix"<<endl;
    
    return exposure;
}

/*
 * Build guassian pyramid
 */
vector<Mat> getGuassianPyr(Mat input)
{
    vector<Mat> result;
    
    Mat temp = input;
    Mat down;
    
    for(int i=0; i<=LEVELS; i++)
    {
        result.push_back(temp);
        
        pyrDown(temp, down, Size(temp.cols/2, temp.rows/2));
        temp = down;
        
        if (temp.cols%2 != 0 || temp.cols/2 == 0 || temp.rows%2 != 0 || temp.rows/2 == 0 || i==LEVELS) {
            cout<<"Last level "<<i<<endl;
            result.push_back(down);
            break;
        }
    }
    
    return result;
}

/*
 * Build Laplacian pyramid
 */
vector<Mat> getLaplacianPyr(Mat input)
{
    vector<Mat> result;
    
    Mat temp = input;
    Mat down, up;
    
    for (int i=0; i<=LEVELS; ++i) {
        
        pyrDown(temp, down, Size(temp.cols/2, temp.rows/2));
        pyrUp(down, up, Size(temp.cols, temp.rows));
        Mat lap = temp - up;
        result.push_back(lap);
        temp = down;
        
        if (temp.cols%2 != 0 || temp.cols/2 == 0 || temp.rows%2 != 0 || temp.rows/2 == 0 || i==LEVELS) {
            cout<<"Last level "<<i<<endl;
            result.push_back(down);
            //down.convertTo(down, CV_32F, 255.0);
            //imwrite("final-down.jpg", down);
            break;
        }
    }
    
    return result;
}

/*
 * Computation of the weight matrix
 */
Mat getWeightMatrix(Mat c, Mat s, Mat e)
{
    Mat weight = cvCreateMat(c.rows, c.cols, CV_32FC1);
    
    for (int i=0; i<c.rows; i++)
    {
        for (int j=0; j<c.cols; j++)
        {
            double cij = c.at<float>(i,j);
            double sij = s.at<float>(i,i);
            double eij = e.at<float>(i,j);
            
            weight.at<float>(i,j) = float(pow(cij,wC)*pow(sij,wS)*pow(eij, wE));
            
        }
    }
    return weight;
}

/*
 * Returns normalised matrix
 * (With respect to the first input, i.e, 'a')
 */
Mat getNormalisedMatrix(Mat a, Mat b)
{
    Mat result = cvCreateMat(a.rows, a.cols, CV_32FC1);
    for (int i=0; i<a.rows; i++)
    {
        for (int j=0; j<a.cols; j++)
        {
            double aij = a.at<float>(i,j);
            double bij = b.at<float>(i,j);
            double nor = aij/(aij+bij);
            if(aij+bij == 0)
                nor = 0.5;
            result.at<float>(i,j) = nor;
        }
    }
    return result;
}

/*
 * Compute combined pyramid
 */
vector<Mat> computeBlenPyr(vector<Mat> gPyr1, vector<Mat> gPyr2, vector<Mat> lPyr1, vector<Mat> lPyr2)
{
    vector<Mat> blend;
    
    for (int i=0; i<gPyr1.size(); i++)
    {
        Mat tempGuas1 = gPyr1[i];
        Mat tempLap1 = lPyr1[i];
        
        Mat tempGuas2 = gPyr2[i];
        Mat tempLap2 = lPyr2[i];
        
        Mat temp = cvCreateMat(tempLap2.rows, tempLap2.cols, CV_32FC3);
        
        for(int i=0; i<tempGuas1.rows; i++)
        {
            for(int j=0; j<tempGuas1.cols; j++)
            {
                double tg1i = tempGuas1.at<float>(i,j);
                double tl1r = tempLap1.at<Vec3f>(i,j)[0];
                double tl1g = tempLap1.at<Vec3f>(i,j)[1];
                double tl1b = tempLap1.at<Vec3f>(i,j)[2];
                
                double tg2i = tempGuas2.at<float>(i,j);
                double tl2r = tempLap2.at<Vec3f>(i,j)[0];
                double tl2g = tempLap2.at<Vec3f>(i,j)[1];
                double tl2b = tempLap2.at<Vec3f>(i,j)[2];
                
                double pixelValueR = tg1i*tl1r + tg2i*tl2r;
                double pixelValueG = tg1i*tl1g + tg2i*tl2g;
                double pixelValueB = tg1i*tl1b + tg2i*tl2b;
                
                temp.at<Vec3f>(i,j)[0] = pixelValueR;
                temp.at<Vec3f>(i,j)[1] = pixelValueG;
                temp.at<Vec3f>(i,j)[2] = pixelValueB;
            }
        }
        blend.push_back(temp);
        //imwrite("temp.jpg", temp);
    }
    
    return blend;
}


/*
 * Collapse the laplacian Pyramid
 */

Mat collapseLaplacianPyr(vector<Mat> src)
{
    Mat result = cvCreateMat(src[src.size()-1].rows, src[src.size()-1].cols, CV_32FC3);
    int i;
    for(i= int(src.size()-1); i>0; i--)
    {
        Mat temp;
        pyrUp(src[i], temp);
        Mat temp2 = src[i-1]+temp;
        src[i-1] = temp2;
    }
    //src[src.size()-1].convertTo(src[src.size()-1], CV_32FC3, 255.0);
    //imwrite("last-level.jpg", src[src.size()-1]);
    src[0].convertTo(src[0], CV_32FC3, 255.0);
    return src[0];
}

