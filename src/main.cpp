#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>
#include <QQuickStyle>
#include <QDebug>
#include "ui/VideoFrameItem.h"
#include "ui/FaceRecognitionBackend.h"
#include "device/CameraManager.h"

int main(int argc, char *argv[])
{
    //全局风格
    QQuickStyle::setStyle("Universal");
    // 使用 QGuiApplication（Qt Quick 推荐）
    QGuiApplication app(argc, argv);

    // 设置应用程序信息
    app.setApplicationName("RK3568 人脸识别系统");
    app.setOrganizationName("RK3568");
    app.setApplicationVersion("2.0");

    // 注册自定义 QML 类型
    qmlRegisterType<VideoFrameItem>("com.rk3568.face", 1, 0, "VideoFrameItem");

    // 创建 QML 引擎
    QQmlApplicationEngine engine;

    // 创建后端实例
    FaceRecognitionBackend backend;

    // 初始化后端（摄像头、模型、数据库）
    if (!backend.initialize()) {
        qCritical() << "后端初始化失败，程序退出";
        return -1;
    }

    // 将后端实例暴露给 QML（作为全局单例）
    engine.rootContext()->setContextProperty("backend", &backend);

    // 注意：由于我们在 QML 中已经有了 VideoFrameItem，
    // 我们需要在 QML 加载后获取它的引用并连接信号
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, [&](QObject *obj, const QUrl &) {
        if (!obj) {
            qCritical() << "QML 加载失败！";
            QGuiApplication::exit(-1);
            return;
        }

        // 强制设置窗口全屏
        QQuickWindow *window = qobject_cast<QQuickWindow*>(obj);
        if (window) {
            window->showFullScreen();
            qDebug() << "✓ 窗口已设置为全屏模式";
        }

        // 查找 VideoFrameItem 实例
        VideoFrameItem *videoFrame = obj->findChild<VideoFrameItem*>("videoFrame");
        if (videoFrame) {
            // 直连相机信号：CameraManager 硬件限速，复用 RGA 转换路径
            QObject::connect(backend.camera(), &CameraManager::newFrameCaptured,
                           videoFrame, &VideoFrameItem::updateFrame);
            qDebug() << "✓ VideoFrameItem 连接成功";
        } else {
            qWarning() << "警告：未找到 VideoFrameItem，摄像头画面可能无法显示";
        }
    });

    // 加载 QML 文件（路径必须匹配 resources.qrc 中的定义）
    const QUrl url(QStringLiteral("qrc:/main.qml"));
    engine.load(url);

    if (engine.rootObjects().isEmpty()) {
        qCritical() << "QML 文件加载失败！";
        return -1;
    }

    qDebug() << "=== 应用程序启动成功 ===";
    return app.exec();
}
