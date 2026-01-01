#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }
    
    Mat frame;

    //重生之我是调库大师（）
    QRCodeDetector qr;

    while (true) {
        
            cap >> frame;
            if (frame.empty()) break;
        

        vector<Point> points;
        string decoded = qr.detectAndDecode(frame, points);
        if (!decoded.empty()) {
            // 绘制边框
            if (points.size() >= 4) {
                for (int j = 0; j < 4; ++j) {
                    line(frame, points[j], points[(j+1)%4], Scalar(0,255,0), 2);
                }
                //  points[(j+1)%4] %3，保证在0-3之间转悠，避免越界访问，小巧思这一块
                putText(frame, decoded, points[0] + Point(0,-10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,0,0), 2);
            } else {
                putText(frame, decoded, Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,0,0), 2);
            }//由于没有配置freetype库，图中无法正常显示中文，只能靠伟大的vsc大人了
            cout << "Decoded: " << decoded << "\n";
        }
        
            imshow("QR Demo", frame);
            if (waitKey(1) == 27) break; // ESC 退出
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
