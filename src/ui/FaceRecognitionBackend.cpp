#include "FaceRecognitionBackend.h"
#include <QDebug>
#include <opencv2/opencv.hpp>

FaceRecognitionBackend::FaceRecognitionBackend(QObject *parent)
    : QObject(parent)
    , m_camera(nullptr)
    , m_retinaface(nullptr)
    , m_mobilefacenet(nullptr)
    , m_facedb(nullptr)
    , m_frameUpdateTimer(nullptr)
    , m_cameraReady(false)
    , m_statusMessage("")
    , m_isProcessing(false)
{
}

FaceRecognitionBackend::~FaceRecognitionBackend()
{
    // 停止定时器
    if (m_frameUpdateTimer) {
        m_frameUpdateTimer->stop();
        delete m_frameUpdateTimer;
    }

    // 安全关闭摄像头线程
    if (m_camera && m_camera->isRunning()) {
        m_camera->closeCamera();
        m_camera->quit();
        m_camera->wait();
    }

    // 释放资源
    if (m_camera) delete m_camera;
    if (m_retinaface) delete m_retinaface;
    if (m_mobilefacenet) delete m_mobilefacenet;
    if (m_facedb) delete m_facedb;
}

bool FaceRecognitionBackend::initialize()
{
    qDebug() << "=== 开始初始化 FaceRecognitionBackend ===";

    // ---------------------------------------------------------
    // 1. 初始化摄像头
    // ---------------------------------------------------------
    m_camera = new CameraManager(this);

    if (m_camera->openCamera(9)) {
        m_cameraReady = true;
        qDebug() << "✓ 摄像头线程已启动 (/dev/video9)";
        emit cameraReadyChanged();
    } else {
        qWarning() << "✗ 摄像头打开失败！";
        m_cameraReady = false;
        setStatusMessage("摄像头打开失败");
        emit cameraReadyChanged();
        return false;
    }

    // 启动定时器，定期从摄像头获取帧并发送到 QML
    m_frameUpdateTimer = new QTimer(this);
    connect(m_frameUpdateTimer, &QTimer::timeout, this, &FaceRecognitionBackend::updateCameraFrame);
    m_frameUpdateTimer->start(33); // 约 30 FPS

    // ---------------------------------------------------------
    // 2. 初始化 RetinaFace 模型
    // ---------------------------------------------------------
    std::string retinaModelPath = "assets/model/retinaface_320.rknn";
    m_retinaface = new RetinaFace(retinaModelPath);

    int retRet = m_retinaface->init();
    if (retRet != 0) {
        qCritical() << "✗ RetinaFace 模型加载失败! 错误码:" << retRet;
        qCritical() << "   路径:" << QString::fromStdString(retinaModelPath);
        setStatusMessage("人脸检测模型加载失败");
        return false;
    } else {
        qDebug() << "✓ RetinaFace 模型加载成功!";
    }

    // ---------------------------------------------------------
    // 3. 初始化 MobileFaceNet 模型
    // ---------------------------------------------------------
    std::string mfnPath = "assets/model/w600k_mbf.rknn";
    m_mobilefacenet = new MobileFaceNet();

    int mfnRet = m_mobilefacenet->init(mfnPath);
    if (mfnRet != 0) {
        qCritical() << "✗ MobileFaceNet 模型加载失败! 错误码:" << mfnRet;
        qCritical() << "   路径:" << QString::fromStdString(mfnPath);
        setStatusMessage("特征提取模型加载失败");
        return false;
    } else {
        qDebug() << "✓ MobileFaceNet 模型加载成功!";
    }

    // ---------------------------------------------------------
    // 4. 初始化人脸数据库
    // ---------------------------------------------------------
    std::string dbPath = "face_database.db";
    m_facedb = new FaceDatabase();

    int dbRet = m_facedb->init(dbPath);
    if (dbRet != 0) {
        qCritical() << "✗ 人脸数据库初始化失败! 错误码:" << dbRet;
        setStatusMessage("数据库初始化失败");
        return false;
    } else {
        int faceCount = m_facedb->getFaceCount();
        qDebug() << "✓ 人脸数据库初始化成功! 当前人脸数量:" << faceCount;
    }

    // ---------------------------------------------------------
    // 初始化完成
    // ---------------------------------------------------------
    setStatusMessage("系统就绪");
    qDebug() << "=== FaceRecognitionBackend 初始化完成 ===";
    return true;
}

void FaceRecognitionBackend::updateCameraFrame()
{
    if (!m_camera) return;

    cv::Mat frame;
    if (m_camera->getLatestFrame(frame) && !frame.empty()) {
        // 将 cv::Mat 转换为 QImage
        QImage image;
        if (frame.type() == CV_8UC3) {
            image = QImage(
                frame.data,
                frame.cols,
                frame.rows,
                static_cast<int>(frame.step),
                QImage::Format_BGR888
            ).copy();
        }

        if (!image.isNull()) {
            emit frameReady(image);
        }
    }
}

void FaceRecognitionBackend::enrollFace()
{
    qDebug() << ">>> 开始执行人脸录入流程...";
    setIsProcessing(true);

    // 1. 检查组件是否就绪
    if (!m_retinaface || !m_mobilefacenet || !m_facedb) {
        qWarning() << "错误：模型或数据库未初始化！";
        setStatusMessage("系统初始化异常");
        setIsProcessing(false);
        return;
    }

    // 2. 从摄像头获取对齐人脸 + 关键点
    qDebug() << "  → 正在从摄像头获取人脸图像...";
    cv::Point2f landmarks[5];
    cv::Mat alignedFace = m_retinaface->getAlignedFaceFromCamera(*m_camera, landmarks);

    if (alignedFace.empty()) {
        qDebug() << "  ✗ 未检测到人脸";
        setStatusMessage("未检测到人脸，请正对摄像头");
        setIsProcessing(false);
        return;
    }
    qDebug() << "  ✓ 成功检测到人脸! 尺寸:" << alignedFace.cols << "x" << alignedFace.rows;

    // 3. 检查是否为正脸
    bool isFrontal = m_retinaface->isFrontalFace(landmarks);
    if (!isFrontal) {
        qDebug() << "  ✗ 不是正脸，请正对摄像头";
        setStatusMessage("请正对摄像头，保持头部水平");
        setIsProcessing(false);
        return;
    }
    qDebug() << "  ✓ 正脸检测通过!";

    // 4. 提取特征向量
    qDebug() << "  → 正在提取人脸特征...";
    std::vector<float> feature;
    int ret = m_mobilefacenet->extractFeature(alignedFace, feature);

    if (ret != 0 || feature.empty()) {
        qCritical() << "  ✗ 特征提取失败，错误码:" << ret;
        setStatusMessage("特征提取失败，请重试");
        setIsProcessing(false);
        return;
    }
    qDebug() << "  ✓ 特征提取成功！维度：" << feature.size();

    // 5. 存入数据库
    qDebug() << "  → 正在将特征存入数据库...";
    int faceId = -1;
    bool enrolled = m_facedb->enrollFace(feature, faceId);

    if (enrolled) {
        QString message = QString("录入成功，序号: %1").arg(faceId);
        qDebug() << "  ✓" << message;
        setStatusMessage(message);
    } else {
        qDebug() << "  ✗ 人脸录入失败: 重复录入";
        setStatusMessage("请不要重复录入");
    }

    setIsProcessing(false);
    qDebug() << "<<< 人脸录入流程结束\n";
}

void FaceRecognitionBackend::recognizeFace()
{
    qDebug() << ">>> 开始执行人脸识别流程...";
    setIsProcessing(true);

    // 1. 检查组件是否就绪
    if (!m_retinaface || !m_mobilefacenet || !m_facedb) {
        qWarning() << "错误：模型或数据库未初始化！";
        setStatusMessage("系统初始化异常");
        setIsProcessing(false);
        return;
    }

    // 2. 从摄像头获取对齐人脸
    qDebug() << "  → 正在从摄像头获取人脸图像...";
    cv::Mat alignedFace = m_retinaface->getAlignedFaceFromCamera(*m_camera);

    if (alignedFace.empty()) {
        qDebug() << "  ✗ 未检测到人脸";
        setStatusMessage("未检测到人脸，请正对摄像头");
        setIsProcessing(false);
        return;
    }
    qDebug() << "  ✓ 成功检测到人脸! 尺寸:" << alignedFace.cols << "x" << alignedFace.rows;

    // 3. 提取特征向量
    qDebug() << "  → 正在提取人脸特征...";
    std::vector<float> feature;
    int ret = m_mobilefacenet->extractFeature(alignedFace, feature);

    if (ret != 0 || feature.empty()) {
        qCritical() << "  ✗ 特征提取失败，错误码:" << ret;
        setStatusMessage("特征提取失败，请重试");
        setIsProcessing(false);
        return;
    }
    qDebug() << "  ✓ 特征提取成功！维度：" << feature.size();

    // 4. 在数据库中查找匹配
    qDebug() << "  → 正在数据库中查找匹配人脸...";
    int faceId = -1;
    bool recognized = m_facedb->recognizeFace(feature, faceId);

    if (recognized) {
        QString message = QString("识别成功：你是 %1 号").arg(faceId);
        qDebug() << "  ✓" << message;
        setStatusMessage(message);
    } else {
        qDebug() << "  ✗ 人脸识别失败: 未找到匹配";
        setStatusMessage("未找到匹配，请先录入人脸");
    }

    setIsProcessing(false);
    qDebug() << "<<< 人脸识别流程结束\n";
}

void FaceRecognitionBackend::setStatusMessage(const QString &message)
{
    if (m_statusMessage != message) {
        m_statusMessage = message;
        emit statusMessageChanged();
        qDebug() << "[状态更新]" << message;
    }
}

void FaceRecognitionBackend::setIsProcessing(bool processing)
{
    if (m_isProcessing != processing) {
        m_isProcessing = processing;
        emit isProcessingChanged();
    }
}
