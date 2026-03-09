# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

基于瑞芯微 RK3568 芯片的实时人脸识别应用。使用 **Qt5 Quick/QML** 构建 UI，通过 RKNN 运行时调用 NPU 硬件加速进行人脸检测（RetinaFace）和特征提取（MobileFaceNet），人脸数据存储在 SQLite 数据库中。

## 构建与部署

### 交叉编译（针对 RK3568，主要工作流）
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/qt5/gcc_arm64 ..
make -j$(nproc)
```

**注意：本项目依赖 RKNN/RGA 硬件库，无法在 x86 本地直接编译运行。**

### 部署到开发板
构建后 `build/deploy/` 文件夹包含完整的运行环境：
```bash
scp -r deploy/ user@rk3568:/opt/faceapp/
ssh user@rk3568
cd /opt/faceapp/deploy && ./RK3568_FaceApp
```

部署包结构：
- `RK3568_FaceApp` - 可执行文件（QML 已静态嵌入）
- `lib/` - RKNN 和 RGA 运行时库（librknnrt.so, librga.so）
- `assets/model/` - RKNN 模型文件（retinaface_320.rknn, w600k_mbf.rknn）
- 数据库文件 `face_database.db` 在首次运行时自动创建

## 核心架构

项目采用**五层架构**：

```
QML 层 (src/qml/) ── 声明式 UI，负责布局和交互
    ↕ Q_PROPERTY / 信号槽
后端桥接层 (FaceRecognitionBackend) ── C++ 与 QML 的桥梁
    ↓
数据库层 (FaceDatabase) ── SQLite 存储特征向量
    ↓
算法层 (RetinaFace, MobileFaceNet) ── RKNN 模型推理
    ↓
设备层 (CameraManager) ── 摄像头捕获（QThread）
```

### 关键数据流

**帧显示流程：**
1. `FaceRecognitionBackend` 内部定时器（33ms）调用 `CameraManager::getLatestFrame()`
2. 将 `cv::Mat` 转换为 `QImage`，通过 `frameReady(QImage)` 信号发送
3. `VideoFrameItem`（QQuickPaintedItem）接收并在 `paint()` 中绘制

**人脸录入流程：**
1. QML 调用 `backend.enrollFace()`
2. `RetinaFace::getAlignedFaceFromCamera()` 获取对齐人脸 + 5 个关键点
3. `RetinaFace::isFrontalFace()` 验证正脸（严格阈值）
4. `MobileFaceNet::extractFeature()` 提取 512 维特征向量
5. `FaceDatabase::enrollFace()` 检查重复并存储到 SQLite
6. 通过 `statusMessage` Q_PROPERTY 通知 QML 更新状态提示

**人脸识别流程：**
1. QML 调用 `backend.recognizeFace()`
2. 同上获取特征向量（识别时不验证正脸）
3. `FaceDatabase::recognizeFace()` 遍历数据库计算余弦相似度
4. 相似度 >= 0.6 返回匹配的人脸 ID

### 各层职责说明

**QML 层 (src/qml/)**
- `main.qml`：主窗口，全屏显示。使用 `anchors` 而非 Layout 进行定位
  - 左侧浅蓝色侧边栏（宽 200px），预留数据库管理区域
  - 右侧主区域：摄像头画面（`VideoFrameItem`）+ 底部按钮
  - 状态提示浮层（`statusHint`）：颜色根据消息类型自动变化
- `ModernButton.qml`：自定义按钮组件，支持按压动画和禁用状态
- **重要**：QML 文件通过 `resources.qrc` 编译嵌入到可执行文件中

**后端桥接层 (src/ui/FaceRecognitionBackend)**
- 继承 `QObject`，通过 `engine.rootContext()->setContextProperty("backend", ...)` 注册为 QML 全局对象
- 暴露给 QML 的 Q_PROPERTY：
  - `cameraReady`（bool）：摄像头是否就绪
  - `statusMessage`（QString）：当前状态文字
  - `isProcessing`（bool）：是否正在处理中（用于禁用按钮）
- 暴露给 QML 的 slots：`enrollFace()`、`recognizeFace()`
- 发送给 `VideoFrameItem` 的信号：`frameReady(QImage)`
- `initialize()` 在 `main.cpp` 中调用，初始化所有子组件

**摄像头显示 (src/ui/VideoFrameItem)**
- 继承 `QQuickPaintedItem`，在 QML 中以 `VideoFrameItem { }` 使用
- 注册方式：`qmlRegisterType<VideoFrameItem>("com.rk3568.face", 1, 0, "VideoFrameItem")`
- `updateFrame(QImage)` slot 接收新帧，调用 `update()` 触发 `paint()`
- `paint()` 按宽高比居中绘制，使用 `SmoothPixmapTransform`
- **渲染目标必须是 `Image`（非 `FramebufferObject`）**，避免嵌入式 GPU 上的 FBO 冲突
- `main.cpp` 通过 `findChild<VideoFrameItem*>("videoFrame")` 找到实例并连接信号

**算法层 (src/algo/)**
- `RetinaFace`：人脸检测（输入 320x320 → 输出人脸框+5个关键点）
  - `getAlignedFaceFromCamera()` / `getAlignedFaceFromCamera(..., landmarks)` - 检测并对齐
  - `isFrontalFace()` - 正脸检测，阈值：Roll 12%，Yaw 12%，对称性 1.20
  - 使用相似变换（Similarity Transform）进行人脸对齐到 112x112
- `MobileFaceNet`：特征提取（输入 112x112 → 输出 512 维向量）
  - `extractFeature()` - 从对齐人脸提取特征向量

**数据库层 (src/db/)**
- `FaceDatabase`：SQLite 封装
  - 表结构：`faces(id INTEGER PRIMARY KEY, feature BLOB)`
  - `enrollFace()` - 录入前检查重复（相似度 >= 0.6 判定为重复）
  - `recognizeFace()` - 遍历所有人脸找最佳匹配
  - **相似度阈值**：`SIMILARITY_THRESHOLD = 0.6f`（FaceDatabase.h）

**设备层 (src/device/)**
- `CameraManager` 继承 QThread，在独立线程中运行捕获循环
- 使用 QMutex 保护 `m_currentFrame` 缓冲区
- 信号 `newFrameCaptured(QImage)` 已保留但主要由 `FaceRecognitionBackend` 的定时器使用

## 线程安全要点

1. **摄像头访问**：必须使用 `CameraManager::getLatestFrame()` 获取帧，内部使用 QMutex 保护
2. **UI 更新**：`FaceRecognitionBackend` 的 `frameReady` 信号通过 Qt 信号槽跨线程发送到 `VideoFrameItem`
3. **RKNN 上下文**：非线程安全，当前实现为单线程推理（在主线程的 Backend slot 中执行）
4. **QML 属性变更**：所有 `emit xxxChanged()` 均在主线程执行，QML 绑定自动更新

## QML 布局注意事项

- **使用 `anchors` 而非 `ColumnLayout`** 进行主区域布局，避免 `fillHeight` 分配问题
- 摄像头区域：`anchors.bottom: buttonRow.top` 确保高度正确
- 按钮行：`anchors.bottom: parent.bottom`，`height: btnHeight`（固定值）
- **禁止在 `VideoFrameItem` 的父 Rectangle 上使用 `layer.enabled: true`**，会导致内容消失（FBO 冲突）
- **禁止 `import QtGraphicalEffects`**（DropShadow 等），在 Mali GPU 上渲染异常
- 在 QML 中检测状态消息类型时使用 `.indexOf()` 替代 `.includes()`（Qt5 兼容性）

## 常见开发场景

### 修改摄像头设备 ID
默认使用 `/dev/video9`，修改位置：
- `src/ui/FaceRecognitionBackend.cpp` 中的 `m_camera->openCamera(9)`

### 调整识别阈值
修改 `src/db/FaceDatabase.h` 中的 `SIMILARITY_THRESHOLD`：
- 增大（如 0.7）→ 更严格，减少误识别
- 减小（如 0.5）→ 更宽松，提高识别率

### 调整正脸检测严格度
修改 `src/algo/RetinaFace.cpp` 中 `isFrontalFace()` 的阈值：
- Roll 阈值：`eye_distance * 0.12f`（第 309 行）
- Yaw 阈值：`eye_distance * 0.12f`（第 320 行）
- 对称性：`symmetry_ratio > 1.20f`（第 338 行）

### 在侧边栏添加数据库管理界面
在 `src/qml/main.qml` 侧边栏的预留区域（标有"即将推出"的 Text 周围）添加新的 QML 组件，并在 `FaceRecognitionBackend` 中添加对应的 Q_INVOKABLE 方法。

### 添加人脸元数据（姓名/时间戳等）
1. 修改 `FaceDatabase::createTable()` 添加字段
2. 更新 `enrollFace()` 和 `recognizeFace()` 的 SQL 和方法签名
3. 在 `FaceRecognitionBackend` 中传递参数
4. 在 QML 中添加输入控件

## 重要文件位置参考

- QML 主界面：`src/qml/main.qml`
- QML 按钮组件：`src/qml/ModernButton.qml`
- 摄像头 QML 显示组件：`src/ui/VideoFrameItem.cpp`（paint 方法）
- C++/QML 桥接：`src/ui/FaceRecognitionBackend.cpp`
  - 录入业务逻辑：`enrollFace()` 方法
  - 识别业务逻辑：`recognizeFace()` 方法
  - 帧定时器：`updateCameraFrame()` 方法
- 人脸检测入口：`src/algo/RetinaFace.cpp`（getAlignedFaceFromCamera）
- 正脸检测：`src/algo/RetinaFace.cpp`（isFrontalFace）
- 特征提取：`src/algo/MobileFaceNet.cpp`（extractFeature）
- 余弦相似度：`src/db/FaceDatabase.cpp`（calculateSimilarity）
- 摄像头线程：`src/device/CameraManager.cpp`（run）
- 应用入口：`src/main.cpp`

## 依赖库

- **Qt5**：Core, Gui, Quick, Qml, QuickControls2（UI 框架，**无 Widgets**）
- **OpenCV**：图像处理和摄像头 I/O
- **SQLite3**：嵌入式数据库
- **RKNN Runtime**：Rockchip NPU 推理引擎（3rdparty/rknn/）
- **RGA Library**：硬件加速图像处理（3rdparty/rga/）

## CMake 配置要点

- 使用 `CMAKE_AUTOMOC/AUTORCC` 处理 Qt 元对象和资源（**无 AUTOUIC**，已无 .ui 文件）
- RPATH 设置为 `$ORIGIN/lib` 实现便携式部署
- QML 文件通过 `resources.qrc` 编译嵌入，运行时无需外部 QML 文件
- POST_BUILD 自动复制库和模型到 `deploy/` 文件夹
- 第三方库路径通过 `RKNN_ROOT` 和 `RGA_ROOT` 变量定义
