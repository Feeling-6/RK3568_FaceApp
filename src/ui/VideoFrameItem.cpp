#include "VideoFrameItem.h"
#include <QPainter>

VideoFrameItem::VideoFrameItem(QQuickItem *parent)
    : QQuickPaintedItem(parent)
{
    // 使用默认 Image 渲染目标（避免与父元素 FBO 冲突）
    setAntialiasing(true);
    setRenderTarget(QQuickPaintedItem::Image);
}

void VideoFrameItem::paint(QPainter *painter)
{
    if (m_image.isNull()) {
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
    m_image = image;
    update();
}
