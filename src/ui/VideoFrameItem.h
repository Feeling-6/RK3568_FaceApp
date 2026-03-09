#ifndef VIDEOFRAMEITEM_H
#define VIDEOFRAMEITEM_H

#include <QQuickPaintedItem>
#include <QImage>
#include <QMutex>
#include <opencv2/opencv.hpp>

/**
 * @brief 自定义 QML 组件，用于显示 OpenCV Mat 格式的摄像头画面
 *
 * 继承 QQuickPaintedItem，在 paint() 方法中将 QImage 绘制到 QML 场景中。
 * 线程安全：使用 QMutex 保护 m_image，因为更新可能来自其他线程。
 */
class VideoFrameItem : public QQuickPaintedItem
{
    Q_OBJECT

public:
    explicit VideoFrameItem(QQuickItem *parent = nullptr);

    /**
     * @brief 重写绘制方法，将当前帧绘制到 QML 场景
     */
    void paint(QPainter *painter) override;

public slots:
    /**
     * @brief 更新要显示的帧（从 QImage）
     * @param image 新的画面帧
     */
    void updateFrame(const QImage &image);

    /**
     * @brief 更新要显示的帧（从 cv::Mat）
     * @param frame OpenCV Mat 格式的画面
     */
    void updateFrameFromMat(const cv::Mat &frame);

private:
    QImage m_image;      // 当前要显示的图像
    QMutex m_mutex;      // 保护 m_image 的互斥锁
};

#endif // VIDEOFRAMEITEM_H
