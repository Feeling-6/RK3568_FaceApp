#ifndef VIDEOFRAMEITEM_H
#define VIDEOFRAMEITEM_H

#include <QQuickPaintedItem>
#include <QImage>

/**
 * @brief 自定义 QML 组件，用于显示摄像头画面
 *
 * 继承 QQuickPaintedItem，在 paint() 方法中将 QImage 绘制到 QML 场景中。
 * 线程安全由 Qt QueuedConnection 保证：updateFrame slot 在主线程执行，
 * QImage 跨线程传递时自动深拷贝，无需手动加锁。
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

private:
    QImage m_image;      // 当前要显示的图像
};

#endif // VIDEOFRAMEITEM_H
