# RK3568 人脸识别系统

基于瑞芯微 **RK3568** 芯片的实时人脸识别应用。利用板载 NPU 进行硬件加速推理，使用 **Qt5 Quick/QML** 构建触摸屏全屏 UI，人脸特征存储于本地 SQLite 数据库。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| 实时摄像头预览 | 30fps 显示，帧由独立线程捕获，UI 线程渲染 |
| 人脸检测 | RetinaFace 模型（320×320 输入），输出人脸框 + 5 个关键点 |
| 正脸验证 | 录入时检查 Roll/Yaw/对称性，拒绝侧脸录入 |
| 人脸对齐 | Similarity Transform 将人脸标准化到 112×112 |
| 特征提取 | MobileFaceNet 提取 512 维特征向量 |
| 人脸录入 | 提取特征后存入 SQLite，录入前去重检测 |
| 人脸识别 | 余弦相似度遍历数据库，阈值 0.6，返回最佳匹配 ID |
| 触摸屏 UI | Qt Quick 全屏界面，左侧边栏预留数据库管理区域 |

---

## 系统架构

### 五层架构

```
┌─────────────────────────────────────────────────────┐
│  QML 层  (src/qml/)                                 │
│  main.qml · ModernButton.qml                        │
│  声明式 UI，负责布局、数据绑定、用户交互            │
└───────────────────┬─────────────────────────────────┘
                    │ Q_PROPERTY 绑定 / 信号槽
┌───────────────────▼─────────────────────────────────┐
│  后端桥接层  (src/ui/)                               │
│  FaceRecognitionBackend · VideoFrameItem            │
│  C++ 与 QML 的桥梁，管理定时器和业务调用流程        │
└──────┬──────────────────────────┬───────────────────┘
       │                          │ frameReady(QImage) 信号
┌──────▼──────────┐    ┌──────────▼──────────────────┐
│  数据库层        │    │  设备层  (src/device/)       │
│  (src/db/)      │    │  CameraManager               │
│  FaceDatabase   │    │  独立 QThread 捕获摄像头帧   │
│  SQLite 特征存储 │    └─────────────────────────────┘
└──────▲──────────┘
       │ 特征向量读写
┌──────▼──────────────────────────────────────────────┐
│  算法层  (src/algo/)                                 │
│  RetinaFace（人脸检测 + 对齐）                       │
│  MobileFaceNet（512维特征提取）                      │
│  NPU 推理由 RKNN Runtime 硬件加速                   │
└─────────────────────────────────────────────────────┘
```

### 关键数据流

**帧显示流程（约 33ms 间隔）：**
```
CameraManager::run()          ← 独立线程持续捕获
    → m_currentFrame (QMutex)
        → FaceRecognitionBackend::updateCameraFrame()  ← QTimer 触发
            → frameReady(QImage) 信号
                → VideoFrameItem::updateFrame()
                    → paint() → 屏幕绘制
```

**人脸录入流程：**
```
QML: backend.enrollFace()
    → RetinaFace::getAlignedFaceFromCamera()   ← 检测 + 对齐到 112×112
        → RetinaFace::isFrontalFace()           ← 正脸验证（拒绝侧脸）
    → MobileFaceNet::extractFeature()           ← 512维特征向量
    → FaceDatabase::enrollFace()                ← 去重检测 + 存入 SQLite
    → setStatusMessage("录入成功 #N")           ← 通知 QML 更新状态提示
```

**人脸识别流程：**
```
QML: backend.recognizeFace()
    → RetinaFace::getAlignedFaceFromCamera()   ← 检测 + 对齐（不验证正脸）
    → MobileFaceNet::extractFeature()           ← 512维特征向量
    → FaceDatabase::recognizeFace()             ← 余弦相似度 >= 0.6 → 匹配
    → setStatusMessage("识别成功 #N" / "未识别") ← 通知 QML
```

---

## 项目结构

```
RK3568_FaceApp/
├── CMakeLists.txt              # 构建配置（Qt5 Quick + OpenCV + RKNN/RGA）
├── resources.qrc               # Qt 资源文件（将 QML 编译嵌入可执行文件）
├── src/
│   ├── main.cpp                # 程序入口，初始化 QML 引擎和后端
│   ├── qml/
│   │   ├── main.qml            # 主界面（全屏，侧边栏 + 摄像头 + 按钮）
│   │   └── ModernButton.qml    # 自定义按钮组件（按压动画、禁用状态）
│   ├── ui/
│   │   ├── FaceRecognitionBackend.h/cpp   # C++/QML 桥接层
│   │   └── VideoFrameItem.h/cpp          # 摄像头帧显示（QQuickPaintedItem）
│   ├── algo/
│   │   ├── RetinaFace.h/cpp    # 人脸检测、对齐、正脸验证
│   │   └── MobileFaceNet.h/cpp # 512维人脸特征提取
│   ├── db/
│   │   └── FaceDatabase.h/cpp  # SQLite 特征向量存取
│   └── device/
│       └── CameraManager.h/cpp # 摄像头捕获线程
├── 3rdparty/
│   ├── rknn/
│   │   ├── include/            # rknn_api.h
│   │   └── lib/                # librknnrt.so
│   └── rga/
│       ├── include/            # im2d.hpp, rga.h
│       └── lib/                # librga.so
└── assets/
    └── model/
        ├── retinaface_320.rknn # RetinaFace 模型
        └── w600k_mbf.rknn      # MobileFaceNet 模型
```

---

## 依赖项

| 库 | 用途 | 获取方式 |
|----|------|----------|
| Qt5 (Core/Gui/Quick/Qml/QuickControls2) | UI 框架 | 交叉编译工具链 |
| OpenCV | 图像处理、摄像头 I/O | 交叉编译 |
| SQLite3 | 特征向量本地存储 | 系统包 / 交叉编译 |
| RKNN Runtime (`librknnrt.so`) | RK3568 NPU 推理引擎 | Rockchip SDK |
| RGA Library (`librga.so`) | 硬件加速图像处理 | Rockchip SDK |

> **注意**：RKNN 和 RGA 是 Rockchip 专有库，**无法在 x86 主机上直接编译运行**，必须交叉编译后部署到 RK3568 板子上。

---

## 构建

### 前置条件

- ARM64 交叉编译工具链（`aarch64-linux-gnu-g++`）
- 适配 RK3568 的 Qt5 交叉编译版本（含 Quick/Qml/QuickControls2）
- 已准备好 `3rdparty/rknn/` 和 `3rdparty/rga/` 目录结构

### 编译步骤

```bash
# 在项目根目录执行
mkdir build && cd build

# 指定 Qt5 交叉编译版本路径
cmake -DCMAKE_PREFIX_PATH=/path/to/qt5/gcc_arm64 \
      -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake \
      ..

make -j$(nproc)
```

编译成功后，`build/deploy/` 目录自动生成完整运行包（由 CMake POST_BUILD 钩子完成）：

```
deploy/
├── RK3568_FaceApp       # 可执行文件（QML 已静态嵌入）
├── lib/
│   ├── librknnrt.so     # RKNN 推理运行时
│   └── librga.so        # RGA 硬件加速库
└── assets/
    └── model/
        ├── retinaface_320.rknn
        └── w600k_mbf.rknn
```

---

## 部署与运行

### 传输到开发板

```bash
scp -r build/deploy/ user@<rk3568-ip>:/opt/faceapp/
```

### 在板端运行

```bash
ssh user@<rk3568-ip>
cd /opt/faceapp/deploy

# 设置显示环境（根据板端实际配置调整）
export DISPLAY=:0
export QT_QPA_PLATFORM=eglfs      # 嵌入式直接渲染（无 X11）
# 或
export QT_QPA_PLATFORM=xcb        # 若有 X11 环境

./RK3568_FaceApp
```

> 数据库文件 `face_database.db` 在首次运行时自动创建于可执行文件同目录。

---

## UI 界面说明

程序启动后自动进入全屏模式：

```
┌──────────┬──────────────────────────────────────┐
│          │                                      │
│  人脸识别 │        摄像头实时画面                 │
│          │                                      │
│  侧边栏   │   （状态提示浮层：成功/失败/处理中）   │
│  (200px) │                                      │
│          │                                      │
│  [预留数  ├──────────────────────────────────────┤
│   据库管  │   [ 录入人脸 ]     [ 识别人脸 ]       │
│   理区域] │                                      │
│          │                                      │
│  ● 摄像头│                                      │
│    就绪   │                                      │
└──────────┴──────────────────────────────────────┘
```

- **状态提示浮层**：显示操作结果，颜色自动变化（绿色=成功，红色=失败，橙色=处理中）
- **按钮禁用状态**：摄像头未就绪或正在处理时，按钮自动变灰不可点击
- **侧边栏预留区**：标注"即将推出"，用于后续添加数据库管理界面

---

## 核心参数调整

### 摄像头设备 ID

默认使用 `/dev/video9`，修改位置：

```cpp
// src/ui/FaceRecognitionBackend.cpp
m_camera->openCamera(9);  // 改为实际设备号
```

### 识别相似度阈值

```cpp
// src/db/FaceDatabase.h
const float SIMILARITY_THRESHOLD = 0.6f;
// 范围建议：0.5（宽松）~ 0.75（严格）
```

### 正脸检测严格度

```cpp
// src/algo/RetinaFace.cpp → isFrontalFace()
// Roll 阈值（眼睛高度差 / 眼距）
float roll_threshold = eye_distance * 0.12f;    // 越小越严格

// Yaw 阈值（鼻/嘴偏移量 / 眼距）
float yaw_threshold  = eye_distance * 0.12f;    // 越小越严格

// 对称性（左眼到鼻距 / 右眼到鼻距）
float symmetry_ratio > 1.20f;                   // 越接近 1.0 越严格
```

### 帧更新频率

```cpp
// src/ui/FaceRecognitionBackend.cpp
m_frameUpdateTimer->start(33);  // 毫秒，约 30fps
```

---

## 技术实现细节

### 为什么用 QQuickPaintedItem 而非 QML Image？

摄像头帧由 C++ 产生（`cv::Mat`），需要通过 Qt 信号槽传递到 QML 渲染层。`VideoFrameItem` 继承 `QQuickPaintedItem`，在 `paint()` 回调中直接用 `QPainter` 绘制 `QImage`，避免了 QML `Image` 需要 `ImageProvider` 的间接层。

**渲染目标必须设为 `Image`**（非默认的 `FramebufferObject`）：

```cpp
setRenderTarget(QQuickPaintedItem::Image);
```

在 RK3568 的 Mali GPU 上，若父 Rectangle 启用 `layer.enabled: true`（FBO 模式），内部的 `VideoFrameItem` 会渲染成白色。使用 `Image` 渲染目标可绕过此冲突。

### 线程安全设计

| 组件 | 线程 | 保护方式 |
|------|------|----------|
| `CameraManager::run()` | 独立 QThread | `QMutex` 保护 `m_currentFrame` |
| `updateCameraFrame()` | 主线程（QTimer）| 无需锁，`getLatestFrame()` 内部加锁 |
| `enrollFace()` / `recognizeFace()` | 主线程（QML slot）| RKNN 上下文非线程安全，串行执行 |
| `VideoFrameItem::paint()` | Qt 渲染线程 | `QMutex` 保护 `m_image` |

### 人脸对齐原理

录入和识别均先执行对齐，确保特征向量与训练数据分布一致：

1. RetinaFace 检测返回 5 个关键点（左眼、右眼、鼻、左嘴角、右嘴角）
2. 与标准人脸模板的 5 个目标点计算 **Similarity Transform**（相似变换矩阵）
3. `cv::warpAffine()` 将原图变换至 112×112 标准人脸图

### QML 与 C++ 的数据绑定

`FaceRecognitionBackend` 通过以下机制与 QML 交互：

```
C++ Q_PROPERTY  ←→  QML 属性绑定（自动更新）
C++ emit signal  →  QML Connections 处理器
QML 调用 backend.enrollFace()  →  C++ public slot
C++ emit frameReady(QImage)  →  VideoFrameItem::updateFrame()
```

---

## 扩展开发指引

### 在侧边栏添加数据库管理界面

1. 在 `src/qml/main.qml` 侧边栏的预留区域替换"即将推出"文字为实际组件
2. 在 `FaceRecognitionBackend` 中添加 `Q_INVOKABLE` 方法（如 `deleteAllFaces()`、`getFaceCount()`）
3. 在 `FaceDatabase` 中已有 `clearAll()` 和 `getFaceCount()` 方法可直接复用

### 添加人脸姓名元数据

1. 修改 `FaceDatabase::createTable()` 中的 SQL：
   ```sql
   CREATE TABLE faces (id INTEGER PRIMARY KEY, name TEXT, feature BLOB)
   ```
2. 更新 `enrollFace()` 和 `recognizeFace()` 方法签名，传入/返回姓名
3. 在 `FaceRecognitionBackend` 的 slot 中增加 `QString name` 参数
4. 在 QML 中添加文本输入框（`TextField` 组件）让用户输入姓名

### 添加新的 QML 页面

由于 QML 文件通过 `resources.qrc` 编译嵌入，添加新文件需：

1. 创建 `src/qml/NewPage.qml`
2. 在 `resources.qrc` 中添加 `<file>src/qml/NewPage.qml</file>`
3. 在 `main.qml` 中通过 `import` 或 `Component` 引用

---

## 已知限制

- **单人脸模式**：每次调用只处理置信度最高的一张人脸，不支持多人同时录入/识别
- **无姓名存储**：数据库仅存储 ID 和特征向量，无法按姓名查询
- **摄像头固定**：设备节点硬编码为 `/dev/video9`
- **x86 不可用**：依赖 RKNN/RGA 硬件库，无法在开发主机上运行

---

## 开发环境

- **目标平台**：Rockchip RK3568，ARM Cortex-A55，Mali-G52 GPU，内置 NPU
- **操作系统**：Linux（Buildroot / Debian 均可）
- **Qt 版本**：Qt 5.15.x
- **编译器**：aarch64-linux-gnu-g++ (GCC 9+)
- **CMake**：3.14+
