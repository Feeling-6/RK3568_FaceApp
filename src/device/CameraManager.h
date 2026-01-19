#ifndef CAMERAMANAGER_H
#define CAMERAMANAGER_H

#include <QObject>
#include <QThread>
#include <QImage>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include "im2d.hpp"
#include "rga.h"

class CameraManager : public QThread // 继承 QThread 以便在后台运行
{
    Q_OBJECT

public:
    explicit CameraManager(QObject *parent = nullptr);
    ~CameraManager();

    // 打开摄像头
    bool openCamera(int deviceId = 0);

    // 关闭摄像头
    void closeCamera();

    /**
     * @brief 供算法层调用，获取当前最新的那一帧图像
     * @param outputFrame 用于接收图像的容器
     * @return 成功返回 true，如果摄像头没开或没数据返回 false
     */
    bool getLatestFrame(cv::Mat& outputFrame);

signals:
    // 信号：通知 UI 层更新画面 (发送 QImage 方便 Qt 显示)
    void newFrameCaptured(const QImage &image);

protected:
    // 线程的主循环函数
    void run() override;

private:
    cv::VideoCapture m_cap;      // OpenCV 视频捕获对象
    cv::Mat m_currentFrame;      // 当前帧 (BGR格式，OpenCV原生)
    bool m_stopThread;           // 线程停止标志位
    QMutex m_mutex;              // 互斥锁，保护 m_currentFrame 的读写安全

    // 辅助函数：将 OpenCV Mat 转换为 Qt QImage
    QImage matToQImage(const cv::Mat &mat);
};

#endif // CAMERAMANAGER_H
