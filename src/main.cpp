/*
 * main.cpp
 *
 *  Created on: Nov 14, 2015
 *      Author: leningli
 */

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

/// Global variables

cv::Mat src, src_gray;
cv::Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  cv::blur( src_gray, detected_edges, cv::Size(3,3) );

  /// Canny detector
  cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = cv::Scalar::all(0);

  src.copyTo( dst, detected_edges);
  cv::imshow( window_name, dst );
 }

void cannyEdgeDetect()
{
    cv::VideoCapture cap(0);                                    //capture the video from web cam

    if ( !cap.isOpened() )                                      // if not success, exit program
    {
         std::cout << "Cannot open the web cam" << std::endl;
         return;
    }

    // Load input image

    while(true)
    {

        bool bSuccess = cap.read(src);                    // read a new frame from video

         if (!bSuccess)                                         //if not success, break loop
        {
             std::cout << "Cannot read a frame from video stream" << std::endl;
             break;
         }

     /// Create a matrix of the same type and size as src (for dst)
     dst.create( src.size(), src.type() );

     /// Convert the image to grayscale
     cv::cvtColor( src, src_gray, CV_BGR2GRAY );

     /// Create a window
     cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );

     /// Create a Trackbar for user to enter threshold
     cv::createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

     /// Show the image
     CannyThreshold(0, 0);

     if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
     {
         std::cout << "esc key is pressed by user" << std::endl;
         break;
     }
 }

}


void colorDetect( )
{
    cv::VideoCapture cap(0);                                    //capture the video from web cam

    if ( !cap.isOpened() )                                      // if not success, exit program
    {
         std::cout << "Cannot open the web cam" << std::endl;
         return;
    }

    // Load input image

    while(true)
    {
        cv::Mat bgr_image;

        bool bSuccess = cap.read(bgr_image);                    // read a new frame from video

         if (!bSuccess)                                         //if not success, break loop
        {
             std::cout << "Cannot read a frame from video stream" << std::endl;
             break;
        }

        // Check if the image can be loaded
        //check_if_image_exist(bgr_image, path_image);

        cv::Mat orig_image = bgr_image.clone();

        cv::medianBlur(bgr_image, bgr_image, 3);

        // Convert input image to HSV
        cv::Mat hsv_image;
        cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);

        // Threshold the HSV image, keep only the black pixels
        cv::Mat lower_black_hue_range;
        cv::Mat upper_black_hue_range;
        cv::inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_black_hue_range);
        cv::inRange(hsv_image, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_black_hue_range);

        // Combine the above two images
        cv::Mat black_hue_image;
        cv::addWeighted(lower_black_hue_range, 1.0, upper_black_hue_range, 1.0, 0.0, black_hue_image);

        cv::GaussianBlur(black_hue_image, black_hue_image, cv::Size(9, 9), 2, 2);

        // Use the Hough transform to detect circles in the combined threshold image
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(black_hue_image, circles, CV_HOUGH_GRADIENT, 1, black_hue_image.rows/8, 100, 20, 0, 0);

        // Loop over all detected circles and outline them on the original image

        for(size_t current_circle = 0; current_circle < circles.size(); ++current_circle) {
            cv::Point center(std::round(circles[current_circle][0]), std::round(circles[current_circle][1]));
            int radius = std::round(circles[current_circle][2]);

            cv::circle(orig_image, center, radius, cv::Scalar(0, 255, 0), 5);
        }

        // Show images
        cv::namedWindow("Threshold lower image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Threshold lower image", lower_black_hue_range);
        cv::namedWindow("Threshold upper image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Threshold upper image", upper_black_hue_range);
        cv::namedWindow("Combined threshold images", cv::WINDOW_AUTOSIZE);
        cv::imshow("Combined threshold images", black_hue_image);
        cv::namedWindow("Detected black circles on the input image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Detected black circles on the input image", orig_image);

        if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

}

static void goodMatchingPoints(cv::Mat descriptors_1 , cv::Mat descriptors_2, std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& good_matches )
{
    double                      max_dist = 0;
    double                      min_dist = 100;

    good_matches.clear();

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    std::cout << " -- Max dist : " << max_dist << std::endl;
    std::cout << " -- Min dist : " << min_dist << std::endl;

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
    //-- PS.- radiusMatch can also be used here.

    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance < 2 * min_dist) {
            good_matches.push_back(matches[i]);
        }
    }
//    matcher.radiusMatch(descriptors_2, descriptors_1, good_matches, 10 );
}

void getMatches(cv::BFMatcher &m_matcher, const cv::Mat &trainDescriptors, const cv::Mat& queryDescriptors,  std::vector<cv::DMatch>& good_matches)
{
    std::vector<std::vector<cv::DMatch> > m_knnMatches;
    good_matches.clear();

    // To avoid NaNs when best match has
    // zero distance we will use inverse ratio.
    const float minRatio =1.f / 1.5f;
    // KNN match will return 2 nearest
    // matches for each query descriptor
    m_matcher.knnMatch(trainDescriptors, queryDescriptors, m_knnMatches,10);
    for (size_t i=0; i < m_knnMatches.size(); i++)
    {
        const cv::DMatch& bestMatch = m_knnMatches[i][0];
        const cv::DMatch& betterMatch = m_knnMatches[i][1];
        float distanceRatio = bestMatch.distance /
            betterMatch.distance;
        // Pass only matches where distance ratio between
        // nearest matches is greater than 1.5
        // (distinct criteria)
        if (distanceRatio < minRatio)
        {
            good_matches.push_back(bestMatch);
        }
    }
}

void KeyPointsToPoints(std::vector<cv::KeyPoint> kpts, std::vector<cv::Point2f> &pts) {
    for (unsigned i = 0; i < kpts.size(); i++) {
        pts.push_back(kpts[i].pt);
    }

    return;
}

bool refineMatchesWithHomography(
        const std::vector<cv::KeyPoint>& queryKeypoints,
        const std::vector<cv::KeyPoint>& trainKeypoints,
        float reprojectionThreshold, std::vector<cv::DMatch>& matches,
        cv::Mat& homography) {
    const unsigned minNumberMatchesAllowed = 8;

    if (matches.size() < minNumberMatchesAllowed)
        return false;

    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());

    for (size_t i = 0; i < matches.size(); i++) {
        srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }

    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC,
            reprojectionThreshold, inliersMask);

    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }

    matches.swap(inliers);
    return matches.size() > minNumberMatchesAllowed;
}

int interestPointsVideoDetect()
{
    cv::Mat img_1;
    img_1 = cv::imread("../images/5.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

    cv::VideoCapture cap(0);                                    //capture the video from web cam

    if ( !cap.isOpened() )                                      // if not success, exit program
    {
         std::cout << "Cannot open the web cam" << std::endl;
         return -1;
    }

    if( ! img_1.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    cv::SurfFeatureDetector     detector( minHessian );
    std::vector<cv::KeyPoint>   keypoints_1, keypoints_2;
    cv::Mat                     img_keypoints_1, img_keypoints_2;

    detector.detect( img_1, keypoints_1 );

    drawKeypoints( img_1, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

//    cv::namedWindow("Keypoints of image 1", cv::WINDOW_AUTOSIZE);

    //-- Step 2: Calculate descriptors (feature vectors)
    cv::SurfDescriptorExtractor extractor;
    cv::Mat                     descriptors_1 , descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );

    //-- Step 3: Matching descriptor vectors with a brute force matcher
    cv::BFMatcher               matcher(cv::NORM_L2);
    std::vector<cv::DMatch>     matches;
    std::vector<cv::DMatch>     good_matches;
    cv::Mat                     img_matches;

    while (true)
    {
        cv::Mat imgOriginal;

        bool bSuccess = cap.read(imgOriginal);                  // read a new frame from video

         if (!bSuccess)                                         //if not success, break loop
        {
             std::cout << "Cannot read a frame from video stream" << std::endl;
             break;
        }

        detector.detect( imgOriginal, keypoints_2);
        drawKeypoints( imgOriginal, keypoints_2, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//        cv::namedWindow("Keypoints of the frame", cv::WINDOW_AUTOSIZE);
        extractor.compute( imgOriginal, keypoints_2, descriptors_2 );
//        matcher.match( descriptors_1, descriptors_2, matches );

        getMatches(matcher,descriptors_1, descriptors_2, good_matches);

        cv::Mat homo;
        float homographyReprojectionThreshold = 1.0;

        bool homographyFound = refineMatchesWithHomography(
                keypoints_1,keypoints_2,homographyReprojectionThreshold,good_matches,homo);
        cv::drawMatches(img_1, keypoints_1, imgOriginal, keypoints_2, good_matches,
                img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("Matches", img_matches);
//        if(good_matches.size() > threshold)
//        {
//            computeBox();
//            drawBox();
//        }
//        else
//            continue;
        if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }                                        // Wait for a keystroke in the window
    return 0;
}

int main(){
//    interestPointsVideoDetect();
//    colorDetect();
    cannyEdgeDetect();
}



