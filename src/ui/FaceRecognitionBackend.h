#ifndef FACERECOGNITIONBACKEND_H
#define FACERECOGNITIONBACKEND_H

#include <QObject>
#include <QString>
#include "device/CameraManager.h"
#include "algo/RetinaFace.h"
#include "algo/MobileFaceNet.h"
#include "db/FaceDatabase.h"

/**
 * @brief QML 后端桥接类，连接 C++ 业务逻辑和 QML UI
 *
 * 管理摄像头、AI 模型、数据库，并通过 Q_PROPERTY 和信号槽与 QML 交互。
 * 所有属性都可以在 QML 中直接绑定，所有业务方法都通过 Q_INVOKABLE 暴露。
 */
class FaceRecognitionBackend : public QObject
{
    Q_OBJECT

    // QML 可绑定的属性
    Q_PROPERTY(bool cameraReady READ cameraReady NOTIFY cameraReadyChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)
    Q_PROPERTY(bool isProcessing READ isProcessing NOTIFY isProcessingChanged)

public:
    explicit FaceRecognitionBackend(QObject *parent = nullptr);
    ~FaceRecognitionBackend();

    // 初始化所有组件（摄像头、模型、数据库）
    bool initialize();

    // 属性 getter
    bool cameraReady() const { return m_cameraReady; }
    QString statusMessage() const { return m_statusMessage; }
    bool isProcessing() const { return m_isProcessing; }

    // 供 main.cpp 建立直连，无需经过 Backend 中转
    CameraManager* camera() const { return m_camera; }

public slots:
    /**
     * @brief 录入人脸（从 QML 调用）
     */
    void enrollFace();

    /**
     * @brief 识别人脸（从 QML 调用）
     */
    void recognizeFace();

signals:
    // 属性变化通知信号
    void cameraReadyChanged();
    void statusMessageChanged();
    void isProcessingChanged();

private:
    /**
     * @brief 设置状态消息并通知 QML
     */
    void setStatusMessage(const QString &message);

    /**
     * @brief 设置处理状态并通知 QML
     */
    void setIsProcessing(bool processing);

    // 业务组件
    CameraManager *m_camera;
    RetinaFace *m_retinaface;
    MobileFaceNet *m_mobilefacenet;
    FaceDatabase *m_facedb;

    // 状态属性
    bool m_cameraReady;
    QString m_statusMessage;
    bool m_isProcessing;
};

#endif // FACERECOGNITIONBACKEND_H
