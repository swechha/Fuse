//
//  ImageFusion.h
//  Fuse
//
//  Created by Swechha Prakash on 10/11/13.
//  Copyright (c) 2013 Swechha Prakash. All rights reserved.
//

#ifndef __Fuse__ImageFusion__
#define __Fuse__ImageFusion__

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

int init(cv::Mat img1, cv::Mat img2);
void imageFusion(cv::Mat source1, cv::Mat source2, int contrast, int saturation, int exposure);

#endif /* defined(__Fuse__ImageFusion__) */
