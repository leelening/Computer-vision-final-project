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

int color_detect( )
{
    cv::VideoCapture cap(0);                                    //capture the video from web cam

    if ( !cap.isOpened() )                                      // if not success, exit program
    {
         std::cout << "Cannot open the web cam" << std::endl;
         return -1;
    }

    cv::namedWindow("Control", CV_WINDOW_AUTOSIZE);             //create a window called "Control"

    int iLowH = 0;
    int iHighH = 179;

    int iLowS = 0;
    int iHighS = 255;

    int iLowV = 0;
    int iHighV = 255;

    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowH, 179);           //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, 179);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255);           //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255);           //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);

    while (true)
    {
        cv::Mat imgOriginal;

        bool bSuccess = cap.read(imgOriginal);                  // read a new frame from video

         if (!bSuccess)                                         //if not success, break loop
        {
             std::cout << "Cannot read a frame from video stream" << std::endl;
             break;
        }

        cv::Mat imgHSV;

        cv::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV);       //Convert the captured frame from BGR to HSV

        cv::Mat imgThresholded;

        cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        //morphological opening (remove small objects from the foreground)
        cv::erode(imgThresholded, imgThresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
        cv::dilate( imgThresholded, imgThresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

        //morphological closing (fill small holes in the foreground)
        cv::dilate( imgThresholded, imgThresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
        cv::erode(imgThresholded, imgThresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

        cv::imshow("Thresholded Image", imgThresholded); //show the thresholded image
        cv::imshow("Original", imgOriginal); //show the original image

        if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;
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

int interest_points_detect()
{
    cv::Mat img_1, img_2;
    img_1 = cv::imread("/home/leningli/workspace/object-recognition/images/1.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    img_2 = cv::imread("/home/leningli/workspace/object-recognition/images/2.jpg", CV_LOAD_IMAGE_COLOR);

    if((! img_1.data) || ( ! img_2.data) )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    cv::SurfFeatureDetector     detector( minHessian );
    std::vector<cv::KeyPoint>   keypoints_1, keypoints_2;
    cv::Mat                     img_keypoints_1, img_keypoints_2;

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    drawKeypoints( img_1, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    drawKeypoints( img_2, keypoints_2, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

    cv::namedWindow("Keypoints of image 1", cv::WINDOW_AUTOSIZE);
    cv::imshow("Keypoints of image 1", img_keypoints_1);
    cv::namedWindow("Keypoints of image 2", cv::WINDOW_AUTOSIZE);
    cv::imshow("Keypoints of image 2", img_keypoints_2);

    //-- Step 2: Calculate descriptors (feature vectors)
    cv::SurfDescriptorExtractor extractor;
    cv::Mat                     descriptors_1 , descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors with a brute force matcher
    cv::BFMatcher               matcher(cv::NORM_L2);
    std::vector<cv::DMatch>     matches;
    std::vector<cv::DMatch>     good_matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    goodMatchingPoints(descriptors_1, descriptors_2, matches, good_matches);

    //-- Draw only "good" matches
    cv::Mat img_matches;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
            img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Show detected matches
    cv::imshow("Good Matches", img_matches);

    cv::waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
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
    for (int i = 0; i < kpts.size(); i++) {
        pts.push_back(kpts[i].pt);
    }

    return;
}

bool refineMatchesWithHomography(
        const std::vector<cv::KeyPoint>& queryKeypoints,
        const std::vector<cv::KeyPoint>& trainKeypoints,
        float reprojectionThreshold, std::vector<cv::DMatch>& matches,
        cv::Mat& homography) {
    const int minNumberMatchesAllowed = 8;

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

int interest_points_video_detect()
{
    cv::Mat img_1;
    img_1 = cv::imread("/home/leningli/workspace/object-recognition/images/4.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

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

    cv::namedWindow("Keypoints of image 1", cv::WINDOW_AUTOSIZE);

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
        cv::namedWindow("Keypoints of the frame", cv::WINDOW_AUTOSIZE);
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
    interest_points_video_detect();
}

//3 General Steps for Object Recognition
//1. Interest Point Detection
//2. Interest Point Description Feature Vector Extraction
//3. Feature Vector Matching Between Two Images

//Load training image
//Detect training interest points
//Extract training interest point descriptors
//Initailize match object
//Initialize and open camera feed
//While(Not User Exit)
//    Grab video frame
//    Detect interest points
//    Extract descriptors
//    Match Query points with training points
//    If(Matching Points > Threshold)
//        Compute Homography Transform Box
//        Draw Box Object and Display
//    Else
//End While

