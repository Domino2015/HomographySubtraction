#include <iostream>
#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/xfeatures2d.hpp>
#include<math.h>

using namespace cv;  //包含cv命名空间
using namespace std;
//融合框架
vector<DMatch> ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint);
Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri);
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
void transform(Mat &SrcImage,Mat &dst);
Mat guidedFilter(Mat &srcMat, Mat &guidedMat, int radius, double eps);//引导滤波器
//将单通道图像转为三通道图像
Mat convertTo3Channels(const Mat& binImg);

int main() {

    //读入图片
    Mat img_1 = imread("../501.png");
    Mat img_2 = imread("../502.png");

    imshow("img_1", img_1);
    //imshow("img_2", img_2);

    //创建SIFT算子
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

    //Detect the keypoints
    vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    //Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);
    f2d->compute(img_2, keypoints_2, descriptors_2);

    //Matching descriptor vector using BFMatcher
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    cout << "SIFT后一共：" << matches.size() << " 对匹配 \n" << endl;

    //绘制粗匹配出的关键点
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    //imshow("RANSC_before", img_matches);
    //TODO:这个是是否开启RANSC,具体可以考虑一下？

    //RANSC误匹配点剔除
    matches = ransac(matches, keypoints_1, keypoints_2);
    Mat img_matches1;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches1);
    //imshow("RANSC_after", img_matches1);


    //获得匹配特征点，并提取最优配对
    sort(matches.begin(),matches.end()); //特征点排序

    //获取排在前N个的最优匹配特征点
    vector<Point2f> imagePoints1,imagePoints2;
    for(int i=0;i<matches.size();i++)
    {
        imagePoints1.push_back(keypoints_1[matches[i].queryIdx].pt);
        imagePoints2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
    //cout<<"点个数："<<imagePoints1.size()<<endl;


    //获取图像1到图像2的投影映射矩阵 尺寸为3*3
    Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);
    cout<<"【1】->【2】变换矩阵为：\n"<<homo<<endl; //输出映射矩阵

    //获取图像2到图像1的投影映射矩阵 尺寸为3*3
    Mat homo1=findHomography(imagePoints2,imagePoints1,CV_RANSAC);
    cout<<"【2】->【1】变换矩阵为：\n"<<homo1<<endl; //输出映射矩阵


    //由图像1到图像2的投影映射矩阵，求得变换后的四个点
    std::vector<Point2f> imagePoints1_after(imagePoints1.size());
    perspectiveTransform( imagePoints1, imagePoints1_after, homo);
    //cout<<"imagePoints1_after"<<imagePoints1_after<<endl;

    //由图像2到图像1的投影映射矩阵，求得变换后的四个点
    std::vector<Point2f> imagePoints2_after(imagePoints1.size());
    perspectiveTransform( imagePoints2, imagePoints2_after, homo1);
    //cout<<imagePoints2_after<<endl;

    std::vector<Point2f> sub_Homography(imagePoints1.size());
    double sub_Homography_x,sub_Homography_y;
    double sub_Homography_x_max,sub_Homography_y_max;
    for(int i=0;i<imagePoints1.size();i++)
    {
        //TODO:误差分析的函数是否可以选用其他模型？
        sub_Homography_x = sqrt(abs(double(imagePoints2_after[i].x*imagePoints2_after[i].x-imagePoints1_after[i].x*imagePoints1_after[i].x)));
        sub_Homography_y = sqrt(abs(double(imagePoints1_after[i].y*imagePoints1_after[i].y - imagePoints2_after[i].y*imagePoints2_after[i].y)));
        //imagePoints2_after[i];
        //cout<<(float)sub_Homography_x<<endl;
        sub_Homography[i].x=(float)sub_Homography_x;
        sub_Homography[i].y=(float)sub_Homography_y;

        if(sub_Homography_x_max-sub_Homography[i].x<0.00001)
        {
            sub_Homography_x_max = sub_Homography[i].x;
        }

        if(sub_Homography_y_max-sub_Homography[i].y <0.00001)
        {
            sub_Homography_y_max = sub_Homography[i].y;
        }

    }
    //cout<< sub_Homography<<endl;
    cout<<"X坐标轴最大值："<<sub_Homography_x_max<<endl;
    cout<<"Y坐标轴最大值："<<sub_Homography_y_max<<endl;
    //归一化
    std::vector<Point2i> sub_Homography_color(imagePoints1.size());
    for(int i=0;i<imagePoints1.size();i++)
    {
        sub_Homography_color[i].x =(int)round(255*(sub_Homography[i].x/(float)sub_Homography_x_max));
        sub_Homography_color[i].y =(int)round(255*(sub_Homography[i].y/(float)sub_Homography_y_max));

        //sub_Homography[i].x =255*(sub_Homography[i].x/(float)sub_Homography_x_max);
        //sub_Homography[i].y =255*(sub_Homography[i].y/(float)sub_Homography_y_max);
    }
    //cout<<sub_Homography<<endl;
    //cout<<"经过归一化矩阵："<<endl;
    //cout<<sub_Homography_color<<endl;

    //TODO:归一化的点，具体分为两类：sub_Homography_color[i].x 图像1到图像2 ；sub_Homography_color[i].y 图像2到图像1
    for(int i=0;i<imagePoints1.size();i++) {
        circle(img_1, imagePoints1[i], 2, cv::Scalar(sub_Homography_color[i].x,255, 0), 2);
        circle(img_2, imagePoints2[i], 2, cv::Scalar(sub_Homography_color[i].y, 255, 0), 2);
    }
    imshow("[1->2]_sub_Homography",img_1);
    imshow("[2->1]_sub_Homography",img_2);

    Mat M = Mat(img_1.rows, img_1.cols, CV_8UC1,Scalar(17));
    for(int i=0;i<imagePoints1.size();i++) {

        M.at<uchar>(imagePoints1[i].y+1, imagePoints1[i].x)= sub_Homography_color[i].x;
        M.at<uchar>(imagePoints1[i].y-1, imagePoints1[i].x)= sub_Homography_color[i].x;
        M.at<uchar>(imagePoints1[i].y, imagePoints1[i].x+1)= sub_Homography_color[i].x;
        M.at<uchar>(imagePoints1[i].y, imagePoints1[i].x-1)= sub_Homography_color[i].x;
        M.at<uchar>(imagePoints1[i].y+1, imagePoints1[i].x+1)= sub_Homography_color[i].x;
        M.at<uchar>(imagePoints1[i].y+1, imagePoints1[i].x-1)= sub_Homography_color[i].x;
        M.at<uchar>(imagePoints1[i].y-1, imagePoints1[i].x+1)= sub_Homography_color[i].x;
        M.at<uchar>(imagePoints1[i].y-1, imagePoints1[i].x-1)= sub_Homography_color[i].x;
    }
    //cout<<M<<endl;

    Mat dst_gauss;
    GaussianBlur(M,dst_gauss,Size(13,13),0,0);
    Mat dst_gauss_3 = convertTo3Channels(dst_gauss);
    //imshow("gauss",dst_gauss_3);
    imwrite("../sub_output.jpg",dst_gauss);
    cv::Mat im_color;
    cv::applyColorMap(dst_gauss_3, im_color, cv::COLORMAP_HSV);
    imshow("color",im_color);

/*
    Mat dst_guidedFilters;
    Mat cloneImage = M.clone();
    dst_guidedFilters = guidedFilter(M,cloneImage,5,0.01);
    imshow("guidedFilters",dst_guidedFilters);
*/
    /*
    //imshow("finish",M);

    cv::Mat src =imread("../501.png");
    Mat im_color;
    cv::applyColorMap(src, im_color, cv::COLORMAP_JET);
    imshow("color",im_color);
*/
    //等待任意按键按下
    waitKey(0);
    return 0;
}

//将单通道图像转为三通道图像
Mat convertTo3Channels(const Mat& binImg)
{
    Mat three_channel = Mat::zeros(binImg.rows,binImg.cols,CV_8UC3);
    vector<Mat> channels;
    for (int i=0;i<3;i++)
    {
        channels.push_back(binImg);
    }
    merge(channels,three_channel);
    return three_channel;
}


//RANSAC算法实现
vector<DMatch> ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint) {
    //定义保存匹配点对坐标
    vector<Point2f> srcPoints(matches.size()), dstPoints(matches.size());
    //保存从关键点中提取到的匹配点对的坐标
    for (int i = 0; i < matches.size(); i++) {
        srcPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
        dstPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;
    }
    //保存计算的单应性矩阵
    Mat homography;
    //保存点对是否保留的标志
    vector<unsigned char> inliersMask(srcPoints.size());
    //匹配点对进行RANSAC过滤 预值：5
    homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 30, inliersMask);
    //RANSAC过滤后的点对匹配信息
    vector<DMatch> matches_ransac;
    //手动的保留RANSAC过滤后的匹配点对
    for (int i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) {
            matches_ransac.push_back(matches[i]);
            //cout<<"第"<<i<<"对匹配："<<endl;
            //cout<<"queryIdx:"<<matches[i].queryIdx<<"\ttrainIdx:"<<matches[i].trainIdx<<endl;
            //cout<<"imgIdx:"<<matches[i].imgIdx<<"\tdistance:"<<matches[i].distance<<endl;
        }
    }
    cout << "经RANSAC消除误匹配后一共：" << matches_ransac.size() << " 对匹配.\n" << endl;
    //返回RANSAC过滤后的点对匹配信息
    return matches_ransac;
}

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri)
{
    Mat originelP,targetP;
    originelP=(Mat_<double>(3,1)<<originalPoint.x,originalPoint.y,1.0);
    targetP=transformMaxtri*originelP;
    float x=targetP.at<double>(0,0)/targetP.at<double>(2,0);
    float y=targetP.at<double>(1,0)/targetP.at<double>(2,0);
    return Point2f(x,y);
}


void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
    std::vector<float> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(float(i));
    for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(float(j));
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

//导向滤波器
Mat guidedFilter(Mat &srcMat, Mat &guidedMat, int radius, double eps)
{
    //------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------
    srcMat.convertTo(srcMat, CV_64FC1);
    guidedMat.convertTo(guidedMat, CV_64FC1);
    //--------------【1】各种均值计算----------------------------------
    Mat mean_p, mean_I, mean_Ip, mean_II;
    boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));//生成待滤波图像均值mean_p
    boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));//生成引导图像均值mean_I
    boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));//生成互相关均值mean_Ip
    boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));//生成引导图像自相关均值mean_II
    //--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_II - mean_I.mul(mean_I);
    //---------------【3】计算参数系数a、b-------------------
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    //--------------【4】计算系数a、b的均值-----------------
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
    boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
    //---------------【5】生成输出矩阵------------------
    Mat dstImage = mean_a.mul(srcMat) + mean_b;
    return dstImage;
}