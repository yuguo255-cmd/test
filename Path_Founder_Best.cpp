#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <iostream>

using namespace cv;
using namespace std;

//框选迷宫的时候左上角得紧贴着或者压在黑边上，这样能大幅提高入口的识别成功率

struct MazePoint {
    Point pos;//position
    vector<Point> path;//用动态数组储存路径
    
    MazePoint(Point p) : pos(p) { path.push_back(p); }//创建迷宫的起点节点
    MazePoint(Point p, vector<Point> prev_path) : pos(p), path(prev_path) {
        path.push_back(p);//把p点添加到path中
    }
};

//封装成一个类  减少参数传递  保护数据完整性

class MazeSolver {
private:
    Rect maze_roi;
    bool roi_defined;// 检查是否已设置ROI
    
public:
    MazeSolver() : roi_defined(false) {} //检查初始化
    
    void setROI(const Rect& roi) {
        maze_roi = roi;
        roi_defined = true;
    }
    
    Mat extractMaze(const Mat& frame) {//提取迷宫部分的图像 //依旧引用避免拷贝
        if (roi_defined) {
            // 确保ROI在图像范围内
            Rect safe_roi = maze_roi & Rect(0, 0, frame.cols, frame.rows);
            if (safe_roi.width > 0 && safe_roi.height > 0) {
                return frame(safe_roi);
            }
        }
        return Mat();
    }
    
    // 改进的图像预处理，解决“空心”问题
    Mat preprocessMaze(const Mat& maze_region) {
        if (maze_region.empty()) return Mat();
        
        Mat gray, binary, inverted;
        
        cvtColor(maze_region, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);

        // 尝试多种二值化方法，找到最适合的
        //感谢ds,Grok,GPT开源（doge）（已经记不清是从哪个大模型问出来的方法了）
        Mat binary1, binary2, binary3;

        // 方法1: 自适应阈值
        adaptiveThreshold(gray, binary1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
                         THRESH_BINARY, 15, 10);

        // 方法2: 全局阈值
        threshold(gray, binary2, 128, 255, THRESH_BINARY);
        
        // 方法3: 另一种自适应阈值
        adaptiveThreshold(gray, binary3, 255, ADAPTIVE_THRESH_MEAN_C, 
                         THRESH_BINARY, 15, 12);
        
        // 选择黑色像素最多的图像（通常是墙壁为黑色，通路为白色）
        int black_pixels1 = countNonZero(binary1 == 0);
        int black_pixels2 = countNonZero(binary2 == 0);
        int black_pixels3 = countNonZero(binary3 == 0);
        
        // 选择黑色像素最多的
        if (black_pixels1 >= black_pixels2 && black_pixels1 >= black_pixels3) {
            binary = binary1;
            cout << "Using adaptive threshold (Gaussian)" << endl;
        } else if (black_pixels2 >= black_pixels1 && black_pixels2 >= black_pixels3) {
            binary = binary2;
            cout << "Using global threshold" << endl;
        } else {
            binary = binary3;
            cout << "Using adaptive threshold (Mean)" << endl;
        }

        // 形态学操作：填充小孔洞，去除小噪声 
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        
        // 先闭运算填充墙壁中的小缝隙 //闭运算：先膨胀再腐蚀
        morphologyEx(binary, binary, MORPH_CLOSE, kernel, Point(-1,-1), 2);
        
        // 再开运算去除通路中的小噪声 //开运算：先腐蚀再膨胀 
        //要是先开后闭，会把墙上的小缝隙放大，后面膨胀也救不回来
        morphologyEx(binary, binary, MORPH_OPEN, kernel, Point(-1,-1), 1);
        
        return binary;
    }
    
    // 入口出口检测
    bool findEntranceExit(const Mat& binary, Point& entrance, Point& exit) {
        if (binary.empty()) return false;
        
        int rows = binary.rows;//据说使用局部变量比访问成员效率更高
        int cols = binary.cols;//但是我还没学4大件，不懂这些。感谢ds开源了。
        
        // 寻找左上角入口（从左上角开始找第一个白色像素）
        bool found_entrance = false;
        for (int row = 0; row < min(rows, 20) && !found_entrance; row++) {
            for (int col = 0; col < min(cols, 20) && !found_entrance; col++) {
                if (binary.at<uchar>(row, col) == 255) {
                    entrance = Point(col, row);
                    found_entrance = true;
                }
            }
        }
        
        // 寻找右下角出口（从右下角开始找第一个白色像素）
        bool found_exit = false;
        for (int row = rows-1; row >= max(0, rows-20) && !found_exit; row--) {//从最后一行开始向上
            for (int col = cols-1; col >= max(0, cols-20) && !found_exit; col--) {//从最后一列开始向左
                if (binary.at<uchar>(row, col) == 255) {
                    exit = Point(col, row);
                    found_exit = true;
                }
            }
        }
        
        // 如果没找到，尝试在整个第一行和最后一行寻找（后备隐藏能源！）
        if (!found_entrance) {
            for (int col = 0; col < cols && !found_entrance; col++) {
                if (binary.at<uchar>(0, col) == 255) {
                    entrance = Point(col, 0);
                    found_entrance = true;
                    cout << "Entrance found at top: " << entrance << endl;
                }//这里输出文字也算是一种debug的方式了
            }
        }
        
        if (!found_exit) {
            for (int col = cols-1; col >= 0 && !found_exit; col--) {
                if (binary.at<uchar>(rows-1, col) == 255) {
                    exit = Point(col, rows-1);
                    found_exit = true;
                    cout << "Exit found at bottom: " << exit << endl;
                }
            }
        }
        
        return found_entrance && found_exit;
    }
    
    // BFS寻找最短路径
    // 通过引用参数返回入口出口,避免后面绘制路线还得再调用一次找出入口的函数
    //（对，一开始我就是调用了两次，后面会标出来，我并没有删掉）
    vector<Point> findShortestPath(const Mat& binary,Point& entrance, Point& exit) {
        if (!findEntranceExit(binary, entrance, exit)) {
            cout << "Cannot find entrance or exit!" << endl;
            return vector<Point>();
        }
        
        int rows = binary.rows;
        int cols = binary.cols;
        
        vector<vector<bool>> visited(rows, vector<bool>(cols, false));//防止重复访问同一个格子
        queue<MazePoint> q;
        
        // 从入口开始
        q.push(MazePoint(entrance));//将起点加入队列
        visited[entrance.y][entrance.x] = true;//标记为已访问
        
        // 四个方向
        vector<Point> directions = {Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1)};
        
        while (!q.empty()) {
            MazePoint current = q.front();
            q.pop();
            
            // 如果到达出口，返回路径
            if (current.pos == exit) {
                cout << "Path found! Length: " << current.path.size() << endl;
                return current.path;
            }
            
            // 检查四个方向
            for (const auto& dir : directions) {
                Point next_pos = current.pos + dir;
                
                // 检查边界
                if (next_pos.x < 0 || next_pos.x >= cols || 
                    next_pos.y < 0 || next_pos.y >= rows) {
                    continue;
                }
                
                // 检查是否是通路（白色）且未访问过
                if (binary.at<uchar>(next_pos) == 255 && 
                    !visited[next_pos.y][next_pos.x]) {
                    visited[next_pos.y][next_pos.x] = true;
                    q.push(MazePoint(next_pos, current.path));
                }
            }
        }
        
        cout << "No path found!" << endl;
        return vector<Point>();
    }
    
    // 在图像上绘制路径
    void drawPath(Mat& image, const vector<Point>& path, const Point& entrance, const Point& exit) {
        if (path.empty()) {
            putText(image, "NO PATH FOUND", Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            return;
        }
        
        // 绘制路径
        for (size_t i = 0; i < path.size() - 1; i++) {
            line(image, path[i], path[i+1], Scalar(0, 255, 0), 3);
        }
        
        // 标记入口和出口
        circle(image, entrance, 5, Scalar(255, 0, 0), -1); // 蓝色入口
        circle(image, exit, 5, Scalar(0, 0, 255), -1);     // 红色出口
        
        // 显示路径长度
        string info = "Path Length: " + to_string(path.size());
        putText(image, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    }
};

// 全局变量用于手动选择ROI
Rect selection;
bool selecting = false;
bool roi_selected = false;

// 鼠标回调函数用于手动选择ROI
void onMouse(int event, int x, int y, int flags, void* param) {
    static Point origin;//记录按下时的起点
    
    if (event == EVENT_LBUTTONDOWN) {
        selecting = true;
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
    } else if (event == EVENT_MOUSEMOVE && selecting) {//鼠标移动即正在选择
        selection.x = min(x, origin.x);
        selection.y = min(y, origin.y);
        selection.width = abs(x - origin.x);
        selection.height = abs(y - origin.y);//细节取绝对值（bushi）
    } else if (event == EVENT_LBUTTONUP) {//左键释放
        selecting = false;                //结束选择
        // 确保选择区域足够大
        if (selection.width > 50 && selection.height > 50) {
            roi_selected = true;         // 标记ROI已选择
            cout << "ROI selected: " << selection << endl;
        } else {
            cout << "Selection too small, please select a larger area." << endl;
        }
    }
}

int main() {
    MazeSolver solver;
    VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        cout << "Cannot open camera!" << endl;
        return -1;
    }
    
    cout << "Camera opened successfully." << endl;
    cout << "Click and drag to select maze area, then press 's' to start processing." << endl;
    cout << "Press 'r' to reselect, 'q' to quit." << endl;
    
    namedWindow("Select Maze Area", WINDOW_NORMAL);
    setMouseCallback("Select Maze Area", onMouse);
    
    bool processing = false;
    
    while (true) {
        Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            cout << "Failed to capture frame!" << endl;
            break;
        }
        
        Mat display = frame.clone();
        
        // 绘制选择矩形
        if (selecting) {
            rectangle(display, selection, Scalar(0, 255, 0), 2);
        }
        
        // 如果已经选择完成，显示固定矩形
        if (roi_selected && !processing) {
            rectangle(display, selection, Scalar(255, 0, 0), 2);
            putText(display, "Press 's' to start", Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }
        
        imshow("Select Maze Area", display);
        
        char key = waitKey(1);
        
        if (key == 's' && roi_selected) {//按下s，开始找路
            solver.setROI(selection);
            processing = true;
            destroyWindow("Select Maze Area");
            cout << "Starting maze processing..." << endl;
            break;
        } else if (key == 'r') {//重新框选
            roi_selected = false;
            cout << "Reselection enabled." << endl;
        } else if (key == 'q') {
            return 0;
        }
    }
    
    namedWindow("Maze Region", WINDOW_NORMAL);
    namedWindow("Processed Maze", WINDOW_NORMAL);
    namedWindow("Result", WINDOW_NORMAL);
    
    while (processing) {
        Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            cout << "Failed to capture frame!" << endl;
            break;
        }
        
        // 提取迷宫区域
        Mat maze_region = solver.extractMaze(frame);
        
        if (!maze_region.empty()) {
            // 预处理迷宫图像
            Mat processed_maze = solver.preprocessMaze(maze_region);
            
            // 寻找最短路径
            Point entrance, exit;
            vector<Point> path = solver.findShortestPath(processed_maze, entrance, exit);
            
            // 在迷宫区域上绘制路径
            Mat result_maze;
            cvtColor(processed_maze, result_maze, COLOR_GRAY2BGR);
            //solver.findEntranceExit(processed_maze, entrance, exit);//啊，对，就是这里。
            solver.drawPath(result_maze, path, entrance, exit);
            
            // 显示结果
            imshow("Maze Region", maze_region);
            imshow("Processed Maze", processed_maze);
            imshow("Result", result_maze);
        }
        
        if (waitKey(1) == 'q') {
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}