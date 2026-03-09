#include "VideoFrameItem.h"
#include <QPainter>
#include <QMutexLocker>

VideoFrameItem::VideoFrameItem(QQuickItem *parent)
    : QQuickPaintedItem(parent)
{
    // 使用默认 Image 渲染目标（避免与父元素 FBO 冲突）
    setAntialiasing(true);
    setRenderTarget(QQuickPaintedItem::Image);
}

void VideoFrameItem::paint(QPainter *painter)
{
    QMutexLocker locker(&m_mutex);

    if (m_image.isNull()) {
        // 没有图像时不绘制
        return;
    }

    // 计算缩放后的尺寸，保持宽高比
    QRectF target = boundingRect();
    QSizeF imageSize = m_image.size();

    // 计算缩放比例（保持宽高比，填充整个区域）
    qreal scaleX = target.width() / imageSize.width();
    qreal scaleY = target.height() / imageSize.height();
    qreal scale = qMax(scaleX, scaleY);

    // 计算居中绘制的位置
    qreal scaledWidth = imageSize.width() * scale;
    qreal scaledHeight = imageSize.height() * scale;
    qreal x = (target.width() - scaledWidth) / 2.0;
    qreal y = (target.height() - scaledHeight) / 2.0;

    QRectF drawRect(x, y, scaledWidth, scaledHeight);

    // 启用平滑转换以获得更好的缩放质量
    painter->setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter->setRenderHint(QPainter::Antialiasing, true);

    // 绘制图像
    painter->drawImage(drawRect, m_image);
}

void VideoFrameItem::updateFrame(const QImage &image)
{
    {
        QMutexLocker locker(&m_mutex);
        m_image = image;
    }

    // 触发重绘
    update();
}

void VideoFrameItem::updateFrameFromMat(const cv::Mat &frame)
{
    if (frame.empty()) {
        return;
    }

    // 将 cv::Mat 转换为 QImage
    QImage image;

    if (frame.type() == CV_8UC3) {
        // BGR 格式（OpenCV 默认）
        image = QImage(
            frame.data,
            frame.cols,
            frame.rows,
            static_cast<int>(frame.step),
            QImage::Format_BGR888
        ).copy(); // 深拷贝以避免数据生命周期问题
    } else if (frame.type() == CV_8UC1) {
        // 灰度图
        image = QImage(
            frame.data,
            frame.cols,
            frame.rows,
            static_cast<int>(frame.step),
            QImage::Format_Grayscale8
        ).copy();
    } else if (frame.type() == CV_8UC4) {
        // RGBA 格式
        image = QImage(
            frame.data,
            frame.cols,
            frame.rows,
            static_cast<int>(frame.step),
            QImage::Format_RGBA8888
        ).copy();
    } else {
        // 不支持的格式，尝试转换为 BGR
        cv::Mat converted;
        cv::cvtColor(frame, converted, cv::COLOR_GRAY2BGR);
        image = QImage(
            converted.data,
            converted.cols,
            converted.rows,
            static_cast<int>(converted.step),
            QImage::Format_BGR888
        ).copy();
    }

    updateFrame(image);
}
