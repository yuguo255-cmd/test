#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;
using namespace std::chrono;

//识别率还行（存疑），大抵是红色圆形的判断有点宽了阈值和上下限应该再卡死一点。

//整体判断逻辑是：先找红圆，再去判断角落像素的颜色。具体可以看主函数上面函数中的判断部分。

// ===== 参数区 =====
const int RED_H_LOW = 0, RED_H_HIGH = 20;       // 红色下限1
const int RED_H_LOW2 = 160, RED_H_HIGH2 = 179;  // 红色下限2
const int S_LOW = 50, V_LOW = 40;               // 饱和度、亮度
const double CIRCLE_THRESH = 0.8;               // 圆度阈值
const int AREA_THRESH = 800;                    // 最小轮廓面积
const int DETECT_INTERVAL = 300;               // 检测间隔(ms)

// ================== 红色圆检测 ==================
bool RedCircle(const Mat& src, Rect roi) {   //用来判断有无红色圆形 //Rect roi 感兴趣区域(Region of Interest)
    if (roi.empty() || roi.x < 0 || roi.y < 0 || 
    roi.x + roi.width > src.cols || roi.y + roi.height > src.rows)
    //if语句换行提高可读性，小巧思这一块 //用于边界检查和空检查
    //用引用传入图像Mat，避免拷贝（太大了）
        return false;

    Mat roiMat = src(roi);//从src里面选出roi框给roiMat（不是拷贝）
    Mat hsv;//HSV对光照变化更鲁棒
    cvtColor(roiMat, hsv, COLOR_BGR2HSV);

    Mat mask1, mask2, redMask;
    inRange(hsv, Scalar(RED_H_LOW, S_LOW, V_LOW), Scalar(RED_H_HIGH, 255, 255), mask1);
    inRange(hsv, Scalar(RED_H_LOW2, S_LOW, V_LOW), Scalar(RED_H_HIGH2, 255, 255), mask2);
    bitwise_or(mask1, mask2, redMask);
    //红色在色相环两头，要算两个范围加起来

    morphologyEx(redMask, redMask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
    //MORPH_OPEN 先腐蚀后膨胀 
    //MORPH_ELLIPSE 要检测红色"圆形"，椭圆结构元素最匹配圆形形状（所以选这个形状的核）
    GaussianBlur(redMask, redMask, Size(5,5), 2);

    vector<vector<Point>> contours;
    findContours(redMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //CHAIN_APPROX_SIMPLE 去除冗余的边界点，只保留形状关键点

    for (auto& c : contours) {  //auto：让编译器自动推断类型  c：循环变量名  contours：要遍历的容器
        //太伟大了现代cpp
        double area = contourArea(c);
        double peri = arcLength(c, true);
        if (area < 80 || peri < 20) continue;
        double circularity = 4 * CV_PI * area / (peri * peri);
        if (circularity > CIRCLE_THRESH)
            return true;
    }
    return false;
}

// ================== 最外框的轮廓提取 ==================
struct ContourResult {
    Mat cropImg;
    Rect rect;
};

ContourResult getContours(const Mat& imgDil, Mat& img) {
    ContourResult result;
    vector<vector<Point>> contours; //轮廓点集
    findContours(imgDil, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double maxArea = 0; //记录最大轮廓面积
    Rect bestRect;

    for (auto& c : contours) { //依旧引用避免拷贝
        double area = contourArea(c); //计算面积
        if (area < AREA_THRESH) continue;//如果面积太小，就跳过这个轮廓，直接处理下一个轮廓

        Rect r = boundingRect(c); 
        double asp = (double)r.width / r.height;
        if (asp < 0.5 || asp > 2.0) continue;  // 通过限制宽高比（asp）避免太扁，
                                              //这样不会把其他矩形的东西当成目标
        if (area > maxArea) {
            maxArea = area;
            bestRect = r;
        }
    }

    if (maxArea > 0) {
        result.rect = bestRect;
        result.cropImg = img(bestRect).clone();
        rectangle(img, bestRect, Scalar(0, 255, 0), 3);
    }

    return result;
}

// ================== 角落颜色判断 ==================
void judge(bool hasRedCircle, const Mat& crop) {
    if (crop.empty()) return;

    int w = crop.cols, h = crop.rows;
    int size = 10; // 取10x10区域平均颜色 光凭一个像素点的颜色显然鲁棒性不好

    auto meanColor = [&](Rect r) -> Scalar { // 以引用方式捕获所有外部变量
        r &= Rect(0, 0, w, h);//&=：交集赋值运算符  
        //用于将矩形裁剪到图像边界内，是处理图像ROI时的防护措施，防止访问无效内存区域导致程序崩溃。
        Mat roi = crop(r);
        return mean(roi);
    };

    Scalar leftColor = meanColor(Rect(0, 0, size, size));
    Scalar rightColor = meanColor(Rect(w - size, 0, size, size));

    //这里通过限定BRG各通道的“量”来控制红色的判断 R > 1.3×G 排除黄色区域  R > 1.3×B 排除紫色区域  R > 80 排除暗色区域（赛场可能比较暗，这里可能得往下调一点）
    //用比值来判断可以适应不同光照    如在强光下所有值都高，但比值不变   什么？你问数值怎么得到的？感谢ds开源（）
    bool leftRed = leftColor[2] > 1.3 * leftColor[1] && leftColor[2] > 1.3 * leftColor[0] && leftColor[2] > 80;
    bool rightRed = rightColor[2] > 1.3 * rightColor[1] && rightColor[2] > 1.3 * rightColor[0] && rightColor[2] > 80;

    if (hasRedCircle) {
        if (leftRed) cout << "Left_2" << endl;
        else if (rightRed) cout << "Right_2" << endl;
        else cout << "U_2" << endl; //这里的U算是一种报错，用来调试参数的（
    } else {
        if (leftRed) cout << "Left_1" << endl;
        else if (rightRed) cout << "Right_1" << endl;
        else cout << "U_1" << endl;
    }
}

// ================== 主函数 ==================
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "No camera" << endl;
        return -1;
    }

    auto last_time = steady_clock::now();//时间控制
    Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;//查空帧，避免处理空帧导致程序崩溃

        auto now = steady_clock::now(); 
        // 判断是否达到检测间隔     间隔检测，小巧思这一块
        if (duration_cast<milliseconds>(now - last_time).count() >= DETECT_INTERVAL) {
            Mat gray, blurImg, edges, dil;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            GaussianBlur(gray, blurImg, Size(5,5), 2);
            Canny(blurImg, edges, 50, 150);
            dilate(edges, dil, getStructuringElement(MORPH_RECT, Size(3,3)));

            Mat show = frame.clone();
            ContourResult cRes = getContours(dil, show);
            if (!cRes.cropImg.empty()) {
                bool hasRedCircle = RedCircle(frame, cRes.rect);
                judge(hasRedCircle, cRes.cropImg);
                imshow("Detected", show);
                imshow("Crop", cRes.cropImg);
            } else {
                cout << "No Rect" << endl;
            }

            last_time = now;// 更新检测时间
        }

        imshow("Camera", frame);
        if (waitKey(1) == 27) break;
    }

    return 0;
}
