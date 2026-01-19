#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include "device/CameraManager.h"
#include "algo/RetinaFace.h"
#include "algo/MobileFaceNet.h"
#include "db/FaceDatabase.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 接收 CameraManager 发过来的图片信号
    void updateCameraImage(const QImage &image);
    // 按钮的点击事件（
    void on_btnEntry_clicked();      // 人脸录入
    void on_btnRecognize_clicked();  // 人脸识别


private:
    Ui::MainWindow *ui;
    CameraManager *m_camera;        // 摄像头管理对象指针
    RetinaFace *m_retinaface;       // 人脸检测模型指针
    MobileFaceNet *m_mobilefacenet; // 人脸特征提取模型指针
    FaceDatabase *m_facedb;         // 人脸数据库管理对象指针
};

#endif // MAINWINDOW_H
