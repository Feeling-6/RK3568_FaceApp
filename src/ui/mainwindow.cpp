#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // ---------------------------------------------------------
    // 初始化摄像头
    // ---------------------------------------------------------
    // 1. 设置窗口启动即最大化
    this->showFullScreen();

    // 2. 初始化摄像头管理器
    m_camera = new CameraManager(this);

    // 3. 连接信号与槽
    // 当 m_camera 发出 newFrameCaptured 信号时，触发 this 的 updateCameraImage 函数
    connect(m_camera, &CameraManager::newFrameCaptured,
            this, &MainWindow::updateCameraImage);

    // 4. 打开摄像头并启动线程
    // openCamera(9)  打开USB摄像头/dev/video9
    if (m_camera->openCamera(9)) {
        m_camera->start(); // 必须调用 start()，因为它是 QThread，这样 run() 才会运行
        qDebug() << "摄像头线程已启动";
    } else {
        qDebug() << "摄像头打开失败！";
        ui->cameraLabel->setStyleSheet("color: red;");
        ui->cameraLabel->setText("摄像头打开失败，请检查设备");
    }

    // ---------------------------------------------------------
    // 初始化 RetinaFace 模型
    // ---------------------------------------------------------
    std::string modelPath = "assets/model/retinaface_320.rknn";

    m_retinaface = new RetinaFace(modelPath);
    int ret = m_retinaface->init();
    if (ret != 0) {
        qDebug() << "RetinaFace模型加载失败! 错误码:" << ret;
        qDebug() << "尝试加载路径:" << QString::fromStdString(modelPath);
        ui->promptLabel->setStyleSheet("color: red;"); // 设置为红色
        ui->promptLabel->setText("RetinaFace模型加载失败");
    } else {
        qDebug() << "RetinaFace模型加载成功!";
    }

    // ---------------------------------------------------------
    // 初始化 MobileFaceNet 模型
    // ---------------------------------------------------------
    std::string mfnPath = "assets/model/w600k_mbf.rknn";
    m_mobilefacenet = new MobileFaceNet(); // 创建实例
    int mfnRet = m_mobilefacenet->init(mfnPath); // 调用 init

    if (mfnRet != 0) {
        qDebug() << "MobileFaceNet模型加载失败! 错误码:" << mfnRet;
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("MobileFaceNet模型加载失败");
    } else {
        qDebug() << "MobileFaceNet模型加载成功!";
    }

    // ---------------------------------------------------------
    // 初始化人脸数据库
    // ---------------------------------------------------------
    std::string dbPath = "face_database.db"; // 数据库文件路径
    m_facedb = new FaceDatabase();           // 创建数据库实例

    int dbRet = m_facedb->init(dbPath);      // 初始化数据库
    if (dbRet != 0) {
        qDebug() << "人脸数据库初始化失败! 错误码:" << dbRet;
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("人脸数据库初始化失败");
    } else {
        qDebug() << "人脸数据库初始化成功! 当前人脸数量:" << m_facedb->getFaceCount();
    }
}

MainWindow::~MainWindow()
{
    // 程序退出前，安全关闭摄像头线程
    if (m_camera->isRunning()) {
        m_camera->closeCamera(); // 调用关闭标志位函数
        m_camera->quit();        // 告诉线程退出事件循环
        m_camera->wait();        // 等待线程真正结束，防止崩溃
    }

    // 释放模型内存
    if (m_retinaface) {
        delete m_retinaface;
        m_retinaface = nullptr;
    }

    if (m_mobilefacenet) {
        delete m_mobilefacenet;
        m_mobilefacenet = nullptr;
    }

    // 释放数据库资源
    if (m_facedb) {
        delete m_facedb;
        m_facedb = nullptr;
    }

    delete ui;
}

// 接收并显示图像
void MainWindow::updateCameraImage(const QImage &image)
{
    if (!image.isNull()) {
        // 将 QImage 转换为 QPixmap 才能在 Label 上显示
        QPixmap pix = QPixmap::fromImage(image);

        // 为了适应屏幕大小，我们让图片自动缩放填充 cameraLabel
        // Qt::KeepAspectRatio: 保持比例（不会把人脸拉扁）
        // Qt::KeepAspectRatioByExpanding: 充满整个框（可能会裁掉一点边缘）
        // Qt::SmoothTransformation: 缩放时平滑处理（画质更好，但稍微费一点CPU）
        int w = ui->cameraLabel->width();
        int h = ui->cameraLabel->height();

        ui->cameraLabel->setPixmap(pix.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

// "人脸识别"按钮逻辑
void MainWindow::on_btnRecognize_clicked()
{
    qDebug() << "开始执行人脸识别流程...";

    // 1. 检查模型和数据库是否就绪
    if (!m_retinaface || !m_mobilefacenet || !m_facedb) {
        qDebug() << "错误：模型或数据库未初始化，无法进行识别！";
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("系统初始化异常，请检查");
        return;
    }

    // 2. 从摄像头实时画面中获取对齐好的 112x112 人脸图像
    qDebug() << "正在从摄像头获取人脸图像...";
    cv::Mat alignedFace = m_retinaface->getAlignedFaceFromCamera(*m_camera);

    if (alignedFace.empty()) {
        qDebug() << "未能在当前画面中检测到人脸";
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("未检测到人脸，请正对摄像头");
        return;
    }
    qDebug() << "成功检测到人脸! 尺寸:" << alignedFace.cols << "x" << alignedFace.rows;

    // 3. 调用 MobileFaceNet 提取人脸特征向量
    qDebug() << "正在提取人脸特征...";
    std::vector<float> feature;
    int ret = m_mobilefacenet->extractFeature(alignedFace, feature);

    if (ret != 0 || feature.empty()) {
        qDebug() << "特征提取失败，RKNN推理返回码:" << ret;
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("特征提取失败，请重试");
        return;
    }
    qDebug() << "特征提取成功！维度：" << feature.size();

    // 4. 在数据库中查找匹配的人脸
    qDebug() << "正在数据库中查找匹配人脸...";
    std::string message;
    int faceId = m_facedb->recognizeFace(feature, message);

    // 5. 根据返回结果更新UI提示
    if (faceId > 0) {
        // 识别成功
        qDebug() << "人脸识别成功！" << QString::fromStdString(message);
        ui->promptLabel->setStyleSheet("color: green;");
        ui->promptLabel->setText(QString::fromStdString(message));
    } else {
        // 识别失败（数据库中没有匹配的人脸）
        qDebug() << "人脸识别失败:" << QString::fromStdString(message);
        ui->promptLabel->setStyleSheet("color: orange;");
        ui->promptLabel->setText(QString::fromStdString(message));
    }
}

// "人脸录入"按钮逻辑
void MainWindow::on_btnEntry_clicked()
{
    qDebug() << "开始执行人脸录入流程...";

    // 1. 检查模型和数据库是否就绪
    if (!m_retinaface || !m_mobilefacenet || !m_facedb) {
        qDebug() << "错误：模型或数据库未初始化，无法进行录入！";
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("系统初始化异常，请检查");
        return;
    }

    // 2. 从摄像头实时画面中获取对齐好的 112x112 人脸图像
    // 该函数在 RetinaFace 类中实现，已经包含了检测、对齐和裁剪逻辑
    qDebug() << "正在从摄像头获取人脸图像...";
    cv::Mat alignedFace = m_retinaface->getAlignedFaceFromCamera(*m_camera);

    if (alignedFace.empty()) {
        qDebug() << "未能在当前画面中检测到人脸";
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("未检测到人脸，请靠近并正对摄像头");
        return;
    }
    qDebug() << "成功检测到人脸! 尺寸:" << alignedFace.cols << "x" << alignedFace.rows;

    // 3. 调用 MobileFaceNet 提取人脸特征向量
    qDebug() << "正在提取人脸特征...";
    std::vector<float> feature;
    int ret = m_mobilefacenet->extractFeature(alignedFace, feature);

    if (ret != 0 || feature.empty()) {
        qDebug() << "特征提取失败，RKNN推理返回码:" << ret;
        ui->promptLabel->setStyleSheet("color: red;");
        ui->promptLabel->setText("特征提取失败，请重试");
        return;
    }
    qDebug() << "特征提取成功！维度：" << feature.size();

    // 4. 将特征向量存入数据库
    qDebug() << "正在将特征存入数据库...";
    std::string message;
    int faceId = m_facedb->enrollFace(feature, message);

    // 5. 根据返回结果更新UI提示
    if (faceId > 0) {
        // 录入成功
        qDebug() << "人脸录入成功！" << QString::fromStdString(message);
        ui->promptLabel->setStyleSheet("color: green;");
        ui->promptLabel->setText(QString::fromStdString(message));
    } else {
        // 录入失败（可能是重复录入）
        qDebug() << "人脸录入失败:" << QString::fromStdString(message);
        ui->promptLabel->setStyleSheet("color: orange;");
        ui->promptLabel->setText(QString::fromStdString(message));
    }
}
