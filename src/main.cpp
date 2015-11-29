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

#define MAX_COUNT 250
#define DELAY_T 20
#define PI 3.1415

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

//void opticialFlow()
//{
//    //////////////////////////////////////////////////////////////////////////
//     //image class
//     IplImage* image = 0;

//     //T, T-1 image
//     IplImage* current_Img = 0;
//     IplImage* Old_Img = 0;

//     //Optical Image
//     IplImage * imgA=0;
//     IplImage * imgB=0;


//     //Video Load
//     CvCapture * capture = cvCreateFileCapture("../images/test.mp4"); //cvCaptureFromCAM(0); //cvCreateFileCapture("1.avi");

//     //Window
//     cvNamedWindow( "Origin" );
//     //////////////////////////////////////////////////////////////////////////


//     //////////////////////////////////////////////////////////////////////////
//     //Optical Flow Variables
//     IplImage * eig_image=0;
//     IplImage * tmp_image=0;
//     int corner_count = MAX_COUNT;
//     CvPoint2D32f* cornersA = new CvPoint2D32f[ MAX_COUNT ];
//     CvPoint2D32f * cornersB = new CvPoint2D32f[ MAX_COUNT ];

//     CvSize img_sz;
//     int win_size=20;

//     IplImage* pyrA=0;
//     IplImage* pyrB=0;

//     char features_found[ MAX_COUNT ];
//     float feature_errors[ MAX_COUNT ];
//     //////////////////////////////////////////////////////////////////////////


//     //////////////////////////////////////////////////////////////////////////
//     //Variables for time different video
//     int one_zero=0;
//     int t_delay=0;



//     //Routine Start
//     while(1) {


//      //capture a frame form cam
//      if( cvGrabFrame( capture ) == 0 )
//       break;

//      //Image Create
//      if(Old_Img == 0)
//      {
//       image = cvRetrieveFrame( capture );
//       current_Img = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
//       Old_Img  = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);



//      }



//      if(one_zero == 0 )
//      {
//       if(eig_image == 0)
//       {
//        eig_image = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
//        tmp_image = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
//       }

//       //copy to image class
//       memcpy(Old_Img->imageData, current_Img->imageData, sizeof(char)*image->imageSize );
//       image = cvRetrieveFrame( capture );
//       memcpy(current_Img->imageData, image->imageData, sizeof(char)*image->imageSize );

//       //////////////////////////////////////////////////////////////////////////
//       //Create image for Optical flow
//       if(imgA == 0)
//       {
//        imgA = cvCreateImage( cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
//        imgB = cvCreateImage( cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
//       }

//       //RGB to Gray for Optical Flow
//       cvCvtColor(current_Img, imgA, CV_BGR2GRAY);
//       cvCvtColor(Old_Img, imgB, CV_BGR2GRAY);

//       //
//       cvGoodFeaturesToTrack(imgA, eig_image, tmp_image, cornersA, &corner_count, 0.01, 5.0, 0, 3, 0, 0.04);
//       cvFindCornerSubPix(imgA, cornersA, corner_count, cvSize(win_size, win_size), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03));


//       CvSize pyr_sz = cvSize( imgA->width+8, imgB->height/3 );
//       if( pyrA == 0)
//       {
//        pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1);
//        pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1);
//       }

//       cvCalcOpticalFlowPyrLK( imgA, imgB, pyrA, pyrB, cornersA, cornersB, corner_count, cvSize(win_size, win_size), 5, features_found, feature_errors, cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3), 0);

//       /////////////////////////////////////////////////////////////////////////

//       for(int i=0; i< corner_count; ++i)
//       {

//        if( features_found[i] == 0 || feature_errors[i] > MAX_COUNT )
//         continue;



//        //////////////////////////////////////////////////////////////////////////
//        //Vector Length
//        float fVecLength = sqrt((float)((cornersA[i].x-cornersB[i].x)*(cornersA[i].x-cornersB[i].x)+(cornersA[i].y-cornersB[i].y)*(cornersA[i].y-cornersB[i].y)));
//        //Vector Angle
//        float fVecSetha  = fabs( atan2((float)(cornersB[i].y-cornersA[i].y), (float)(cornersB[i].x-cornersA[i].x)) * 180/PI );

//        cvLine( image, cvPoint(cornersA[i].x, cornersA[i].y), cvPoint(cornersB[i].x, cornersA[i].y), CV_RGB(0, 255, 0), 2);

//        printf("[%d] - Sheta:%lf, Length:%lf\n",i , fVecSetha, fVecLength);
//       }


//       //////////////////////////////////////////////////////////////////////////

//      }
//      cvShowImage( "Origin", image );

//      //////////////////////////////////////////////////////////////////////////

//      //time delay
//    one_zero++;
//      if( (one_zero % DELAY_T ) == 0)
//      {
//       one_zero=0;
//      }

//      //break
//      if( cvWaitKey(10) >= 0 )
//       break;
//     }

//     //release capture point
//     cvReleaseCapture(&capture);
//     //close the window
//     cvDestroyWindow( "Origin" );

//     cvReleaseImage(&Old_Img);
//     //////////////////////////////////////////////////////////////////////////
//     cvReleaseImage(&imgA);
//     cvReleaseImage(&imgB);
//     cvReleaseImage(&eig_image);
//     cvReleaseImage(&tmp_image);
//     delete cornersA;
//     delete cornersB;
//     cvReleaseImage(&pyrA);
//     cvReleaseImage(&pyrB);


//     //////////////////////////////////////////////////////////////////////////
//}
using namespace cv;
using namespace std;

// General settings
#define Z_DEPTH 133.4f // cm
#define FOCAL_LENGTH .00342f // cm
#define LENGTH_PER_PIXEL 0.0000049f // cm
#define ITERATIONS_PER_SEED 5

// Display settings
// Defining "OPTFLOW_DISPLAY" (#define OPTFLOW_DISPLAY) enables graphical output for this application. This normally shouldn't be done in the source code but rather done in the IDE or Makefile as to not interfere with different build methods.
#define OVERLAY_CIRCLE_RADIUS 5
#define OVERLAY_COLOR_R 255
#define OVERLAY_COLOR_B 0
#define OVERLAY_COLOR_G 0

// Shi-Tomasi settings. Used when finding the seed corners
#define SHITOMASI_MAX_CORNERS 100
#define SHITOMASI_QUALITY_LEVEL 0.3f
#define SHITOMASI_MIN_DISTANCE 7
#define SHITOMASI_BLOCK_SIZE 7

// Lucas-Kanad Optical Flow settings. Used to track the seed corners until next seed.
#define LUCASKANAD_WINDOW_SIZE_X 15
#define LUCASKANAD_WINDOW_SIZE_Y 15
#define LUCASKANAD_MAX_LEVEL 2

const Scalar color(OVERLAY_COLOR_B, OVERLAY_COLOR_G, OVERLAY_COLOR_R);
const double multiplier = (Z_DEPTH / FOCAL_LENGTH) * LENGTH_PER_PIXEL;

#define UNKNOWN_FLOW_THRESH 1e9

void makecolorwheel(vector<Scalar> &colorwheel)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;

    int i;

    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));
}

void motionToColor(Mat flow, Mat &color)
{
    if (color.empty())
        color.create(flow.rows, flow.cols, CV_8UC3);

    static vector<Scalar> colorwheel; //Scalar r,g,b
    if (colorwheel.empty())
        makecolorwheel(colorwheel);

    // determine motion range:
    float maxrad = -1;

    // Find max flow to normalize fx and fy
    for (int i= 0; i < flow.rows; ++i)
    {
        for (int j = 0; j < flow.cols; ++j)
        {
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);
            float fx = flow_at_point[0];
            float fy = flow_at_point[1];
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
                continue;
            float rad = sqrt(fx * fx + fy * fy);
            maxrad = maxrad > rad ? maxrad : rad;
        }
    }

    for (int i= 0; i < flow.rows; ++i)
    {
        for (int j = 0; j < flow.cols; ++j)
        {
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);

            float fx = flow_at_point[0] / maxrad;
            float fy = flow_at_point[1] / maxrad;
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
            {
                data[0] = data[1] = data[2] = 0;
                continue;
            }
            float rad = sqrt(fx * fx + fy * fy);

            float angle = atan2(-fy, -fx) / CV_PI;
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);
            int k0 = (int)fk;
            int k1 = (k0 + 1) % colorwheel.size();
            float f = fk - k0;
            //f = 0; // uncomment to see original color wheel

            for (int b = 0; b < 3; b++)
            {
                float col0 = colorwheel[k0][b] / 255.0;
                float col1 = colorwheel[k1][b] / 255.0;
                float col = (1 - f) * col0 + f * col1;
                if (rad <= 1)
                    col = 1 - rad * (1 - col); // increase saturation with radius
                else
                    col *= .75; // out of range
                data[2 - b] = (int)(255.0 * col);
            }
        }
    }
}

float ds_factor = 0.75;

static void help()
{
    cout <<
            "\nDense optical flow algorithm by Gunnar Farneback\n"
            "Usage:\n"
            "$ ./main\n"
            "This reads input from the webcam\n" << endl;
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int opticialFlow()
{
//    // The change in position of the camera
//    float dx, dy;
//    dx = dy = 0;

//    // The change in position for a single iteration
//    float ddx, ddy;

//    Mat cur_frame;
//    Mat cur_fgray;
//    Mat old_fgray;

////#ifdef OPTFLOW_DISPLAY
//    // Optical flow overlay
//    Mat overlay;
////#endif

//    // Corner points being tracked
//    vector<Point2f> old_p, cur_p, found_p;

//    vector<unsigned char> status;
//    vector<float> err;

//    // Lucas Kanad Optical Flow parameters
//    Size winSize(LUCASKANAD_WINDOW_SIZE_X, LUCASKANAD_WINDOW_SIZE_Y);
//    TermCriteria criteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 0.03);

//    VideoCapture cap(-1);
//    if (!cap.isOpened()) {
//        cout << "Failed to open camera.\n";
//        return 1;
//    }

//    while (true) {
//        // Get a new frame
//        cap >> cur_frame;
//        cvtColor(cur_frame, old_fgray, CV_BGR2GRAY);

//        // Get new corners from this frame
//        goodFeaturesToTrack(old_fgray,
//                            old_p,
//                            SHITOMASI_MAX_CORNERS,
//                            SHITOMASI_QUALITY_LEVEL,
//                            SHITOMASI_MIN_DISTANCE,
//                            noArray(),
//                            SHITOMASI_BLOCK_SIZE);

//        // If we didn't get get feature to track, try again
//        if (old_p.size() == 0) {
//            continue;
//        }

////#ifdef OPTFLOW_DISPLAY
//        // Reset the mask
//        overlay = Mat::zeros(cur_frame.size(), cur_frame.type());
////#endif

//        // Track these corners over the next few frames
//        for (int i=0; i<ITERATIONS_PER_SEED; i++) {

//            // Get a new frame
//            cap >> cur_frame;
//            cvtColor(cur_frame, cur_fgray, CV_BGR2GRAY);

//            // Track corners over the current frame
//            calcOpticalFlowPyrLK(old_fgray,
//                                 cur_fgray,
//                                 old_p,
//                                 cur_p,
//                                 status,
//                                 err,
//                                 winSize,
//                                 LUCASKANAD_MAX_LEVEL,
//                                 criteria);

//            found_p.clear();
//            ddx = ddy = 0;

//            // Iterate over the corners
//            for (int k=0; k<status.size(); k++) {
//                if (status[k] > 0) {
//                    // If the status value is 1 then this corner was found so we add it to the found list
//                    found_p.push_back(cur_p[k]);

//                    // Add to the running sum of ddx/ddy
//                    ddx += cur_p[k].x - old_p[k].x;
//                    ddy += cur_p[k].y - old_p[k].y;

////#ifdef OPTFLOW_DISPLAY
//                    // Draw the optical flow lines onto the mask
//                    circle(cur_frame, cur_p[k], OVERLAY_CIRCLE_RADIUS, color, -1);
//                    line(overlay, old_p[k], cur_p[k], color);
////#endif
//                }
//            }

//            // If no corners were tracked this iteration, reseed
//            if (found_p.size() == 0) {
//                break;
//            }

//            // Calculate the change in position for this iteration
//            dx += (ddx / found_p.size()) * multiplier;
//            dy += (ddy / found_p.size()) * multiplier;

//            // Use all the tracked points in the next iteration (points that couldn't be tracked are thrown out)
//            old_p.clear();
//            old_p = found_p;

//            cout << "dx: " << dx << " dy: " << dy << endl;

////#ifdef OPTFLOW_DISPLAY
//            // Display frame
//            add(overlay, cur_frame, cur_frame);
//            imshow("frame", cur_frame);
//            if(waitKey(30) >= 0) break;
////#endif
//        }
//    }

//    return 0;
//    VideoCapture cap;
//        cap.open(0);
//        //cap.open("test_02.wmv");

//        if( !cap.isOpened() )
//            return -1;

//        Mat prevgray, gray, flow, cflow, frame;
//        namedWindow("flow", 1);

//        Mat motion2color;

//        for(;;)
//        {
//            double t = (double)cvGetTickCount();

//            cap >> frame;
//            cvtColor(frame, gray, CV_BGR2GRAY);
//            imshow("original", frame);

//            if( prevgray.data )
//            {
//                calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//                motionToColor(flow, motion2color);
//                imshow("flow", motion2color);
//            }
//            if(waitKey(10)>=0)
//                break;
//            std::swap(prevgray, gray);

//            t = (double)cvGetTickCount() - t;
//            cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
//        }
//        return 0;
    VideoCapture cap(0);
    help();
    if( !cap.isOpened() )
        return -1;

    Mat prevgray, gray, flow, cflow, frame;
    namedWindow("flow", 1);

    for(;;)
    {
        cap >> frame;
        resize(frame, frame, Size(), ds_factor, ds_factor, INTER_NEAREST);
        cvtColor(frame, gray, CV_BGR2GRAY);

        if( prevgray.data )
        {
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            cvtColor(prevgray, cflow, CV_GRAY2BGR);
            drawOptFlowMap(flow, cflow, 16, 1.5, CV_RGB(0, 255, 0));
            imshow("flow", cflow);
        }
        if(waitKey(30)>=0)
            break;
        std::swap(prevgray, gray);
    }
}



int main(){
//    interestPointsVideoDetect();
//    colorDetect();
//    cannyEdgeDetect();
    opticialFlow();
}



