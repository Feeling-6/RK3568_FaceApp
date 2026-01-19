#include "CameraManager.h"
#include <QDebug>

CameraManager::CameraManager(QObject *parent)
    : QThread(parent), m_stopThread(false)
{
}

CameraManager::~CameraManager()
{
    closeCamera();
    wait(); // 等待线程完全退出
}

bool CameraManager::openCamera(int deviceId)
{
    // 如果已经打开，先关闭
    if (m_cap.isOpened()) {
        closeCamera();
    }

    // 打开摄像头设备 /dev/videoX
    m_cap.open(deviceId);
    if (!m_cap.isOpened()) {
        qCritical() << "Error: Cannot open camera" << deviceId;
        return false;
    }

    // --- 关键设置：针对 USB 摄像头的优化 ---
    // 设置为 MJPG 格式，保证高帧率传输
    m_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // 设置分辨率，建议 1280x720，既清晰又比 1080p 跑得快(我就要1080p)
    m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    qDebug() << "Camera opened successfully.";

    // 启动采集线程
    m_stopThread = false;
    start();

    return true;
}

void CameraManager::closeCamera()
{
    m_stopThread = true;
    // 等待 run 函数执行完毕
    if (isRunning()) {
        wait(1000);
    }

    if (m_cap.isOpened()) {
        m_cap.release();
    }
}

// 供算法层调用：线程安全地获取最新帧
bool CameraManager::getLatestFrame(cv::Mat& outputFrame)
{
    QMutexLocker locker(&m_mutex); // 自动加锁，作用域结束自动解锁

    if (m_currentFrame.empty()) {
        return false;
    }

    // 必须使用 clone() 深拷贝，否则外部修改图像会影响这里
    outputFrame = m_currentFrame.clone();
    return true;
}

// 线程主循环：不断读取摄像头并发送信号
void CameraManager::run()
{
    cv::Mat tempFrame;

    while (!m_stopThread) {
        if (!m_cap.read(tempFrame)) {
            qWarning() << "Failed to read frame from camera";
            QThread::msleep(10); // 读不到数据时稍微休息一下，避免 CPU 100%
            continue;
        }

        if (tempFrame.empty()) continue;

        // 1. 存入缓冲区供算法使用 (加锁保护)
        {
            QMutexLocker locker(&m_mutex);
            m_currentFrame = tempFrame;
            // 注意：这里保存的是 BGR 格式，因为 OpenCV 算法通常用 BGR
        }

        // 2. 转换为 QImage 发送给 UI (不加锁，避免阻塞采集)
        QImage image = matToQImage(tempFrame);
        emit newFrameCaptured(image);

        // 稍微休眠一下控制帧率，防止把 UI 线程卡死 (例如限制在 30fps)
        QThread::msleep(16);
    }
}

QImage CameraManager::matToQImage(const cv::Mat &mat)
{
    if (mat.empty()) return QImage();
    
    // 1. 定义源数据 (OpenCV BGR)
    rga_buffer_t src = wrapbuffer_virtualaddr(
        (void*)mat.data, 
        mat.cols, 
        mat.rows, 
        RK_FORMAT_BGR_888
    );

    // 2. 准备目标 QImage
    // 为了降低 UI 压力，这里其实可以顺便把图缩小
    // 比如：如果你的 UI 控件只有 640x360，这里直接生成 640x360 的 QImage 最好
    // 下面演示保持原尺寸 (1280x720)，如果你想缩放，修改这里的 width/height 即可
    int dstWidth = mat.cols;  // 或者 640
    int dstHeight = mat.rows; // 或者 360
    
    // 创建一个空的 QImage 容器，分配内存
    QImage image(dstWidth, dstHeight, QImage::Format_RGB888);

    // 3. 定义目标数据 (QImage 内存, RGB)
    rga_buffer_t dst = wrapbuffer_virtualaddr(
        (void*)image.bits(), 
        dstWidth, 
        dstHeight, 
        RK_FORMAT_RGB_888
    );

    // 4. 执行转换 (Copy + Format Convert + Resize if needed)
    // imresize 可以同时处理 缩放 和 格式转换
    // 如果尺寸一样，它就只做格式转换
    IM_STATUS status = imresize(src, dst);

    if (status != IM_STATUS_SUCCESS) {
        qWarning() << "RGA matToQImage failed:" << status;
        return QImage(); 
    }

    return image;

    /* 原代码使用opencv(CPU 慢速转换):
    if(mat.type() == CV_8UC3) {
        QImage image((const uchar *)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    return QImage();
    */
}
