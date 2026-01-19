# README.md

此文件为 Claude Code (claude.ai/code) 处理本仓库代码时提供指导。[English Version](./README_EN.md)

## 项目概览

基于瑞芯微 RK3568 芯片的实时人脸识别应用，支持人脸检测、录入和识别功能。使用 Qt5、OpenCV 构建，通过 Rockchip RKNN 运行时调用 NPU 硬件加速。

### 核心功能

- **实时人脸检测**: 使用 RetinaFace 模型从摄像头画面检测人脸
- **人脸录入**: 采集并将人脸特征存储到 SQLite 数据库
- **人脸识别**: 通过余弦相似度算法匹配数据库中的人脸
- **硬件加速**: 利用 RK3568 NPU 通过 RKNN 运行时实现快速推理
- **线程安全设计**: 多线程架构保证 UI 流畅和实时处理
- **一键部署**: 单文件夹部署，包含所有依赖

---

## 快速开始

### 编译项目

```bash
mkdir -p build && cd build
cmake ..
make
```

### 部署到 RK3568 开发板

编译完成后，`build/deploy/` 文件夹包含运行所需的所有内容:

```bash
# 将整个 deploy 文件夹复制到开发板
scp -r deploy/ user@rk3568:/path/to/app/

# 在开发板上运行程序
cd /path/to/app/deploy
./RK3568_FaceApp
```

### 使用方法

1. **启动应用**: 程序以全屏模式打开，显示摄像头预览
2. **录入人脸**: 点击"录入"按钮将新人脸添加到数据库
3. **识别人脸**: 点击"识别"按钮识别当前人脸

---

## 项目结构

```
RK3568_FaceApp/
├── assets/
│   └── model/              # RKNN 模型文件
│       ├── retinaface_320.rknn
│       └── w600k_mbf.rknn
├── src/
│   ├── main.cpp            # 应用程序入口
│   ├── device/             # 设备层（摄像头）
│   │   ├── CameraManager.h
│   │   └── CameraManager.cpp
│   ├── algo/               # 算法层（AI 模型）
│   │   ├── RetinaFace.h/cpp
│   │   └── MobileFaceNet.h/cpp
│   ├── db/                 # 数据库层
│   │   ├── FaceDatabase.h
│   │   └── FaceDatabase.cpp
│   └── ui/                 # UI 层
│       ├── mainwindow.h/cpp
│       └── mainwindow.ui
├── 3rdparty/               # 第三方库
│   ├── rknn/               # RKNN 运行时头文件和库
│   └── rga/                # RGA 图像处理库
└── CMakeLists.txt
```

---

## 架构设计

### 四层架构

应用采用清晰的四层架构设计，模块化且易于维护:

```
┌─────────────────────────────────────────────────┐
│              UI 层 (Qt 图形界面)                 │
│  - MainWindow: 显示与用户交互                    │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│           数据库层 (SQLite)                      │
│  - FaceDatabase: 存储与匹配人脸特征              │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│         算法层 (RKNN 模型)                       │
│  - RetinaFace: 人脸检测与对齐                    │
│  - MobileFaceNet: 特征提取                       │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│          设备层 (摄像头 I/O)                      │
│  - CameraManager: 线程安全的帧捕获               │
└─────────────────────────────────────────────────┘
```

### 1. 设备层 (`src/device/`)

**CameraManager** - 基于 QThread 的摄像头捕获

- 使用 OpenCV VideoCapture 打开 `/dev/videoX` 设备
- 在后台线程中连续捕获视频帧
- 通过 Qt 信号 (`newFrameCaptured`) 更新 UI
- 提供线程安全的 `getLatestFrame()` 供算法层访问
- 使用 QMutex 保护帧缓冲区

**关键方法:**
- `openCamera(int deviceId)`: 打开摄像头设备
- `getLatestFrame(cv::Mat& frame)`: 获取当前帧（线程安全）

### 2. 算法层 (`src/algo/`)

#### RetinaFace - 人脸检测

- **输入**: 来自摄像头的 320x320 RGB 图像
- **输出**: 人脸边界框、置信度分数、5 个面部关键点
- **加速**: RKNN（NPU 推理）

**关键方法:**
- `init()`: 加载 RKNN 模型并初始化
- `getAlignedFaceFromCamera(CameraManager& camera)`: 检测人脸并返回对齐的 112x112 图像

**实现细节:**
- 在 `initPriors()` 中预计算锚点 (BOX_PRIORS_320)
- 基于 5 个关键点使用相似变换进行人脸对齐
- 返回对齐并裁剪的 112x112 人脸图像，可直接用于特征提取

#### MobileFaceNet - 特征提取

- **输入**: 112x112 对齐的人脸图像
- **输出**: 512 维特征向量（embeddings）
- **加速**: RKNN（NPU 推理）

**关键方法:**
- `init(const std::string& modelPath)`: 加载 RKNN 模型
- `extractFeature(const cv::Mat& alignedFace, std::vector<float>& feature)`: 提取人脸特征向量

### 3. 数据库层 (`src/db/`)

**FaceDatabase** - 基于 SQLite 的人脸特征存储

- **存储方式**: 人脸特征以 BLOB（二进制大对象）形式存储
- **匹配算法**: 余弦相似度算法进行人脸识别
- **匹配阈值**: 相似度分数 >= 0.6 判定为同一个人

**数据库表结构:**
```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    feature BLOB NOT NULL                   -- 特征向量（二进制）
);
```

**关键方法:**
- `init(const std::string& db_path)`: 初始化 SQLite 数据库
- `enrollFace(const std::vector<float>& feature, std::string& message)`: 添加新人脸（含重复检查）
- `recognizeFace(const std::vector<float>& feature, std::string& message)`: 查找匹配的人脸
- `clearAll()`: 清空所有人脸数据
- `getFaceCount()`: 获取已录入人脸数量

**识别算法:**
- 使用余弦相似度: `cos(θ) = (A·B) / (|A|×|B|)`
- 遍历所有已存储的人脸找到最佳匹配
- 相似度 >= 0.6 则返回人脸 ID，否则返回 -1

### 4. UI 层 (`src/ui/`)

**MainWindow** - 基于 Qt 的图形界面

- 使用 QLabel 显示实时摄像头画面
- 两个主要按钮: "录入" 和 "识别"
- 状态标签使用颜色编码显示操作结果

**UI 工作流程:**

**人脸录入 (`on_btnEntry_clicked`):**
1. 从摄像头获取对齐人脸 → `RetinaFace::getAlignedFaceFromCamera()`
2. 提取特征向量 → `MobileFaceNet::extractFeature()`
3. 存入数据库 → `FaceDatabase::enrollFace()`
4. 显示结果（绿色=成功，橙色=重复，红色=错误）

**人脸识别 (`on_btnRecognize_clicked`):**
1. 从摄像头获取对齐人脸 → `RetinaFace::getAlignedFaceFromCamera()`
2. 提取特征向量 → `MobileFaceNet::extractFeature()`
3. 在数据库中匹配 → `FaceDatabase::recognizeFace()`
4. 显示结果（绿色=识别成功"你是X号"，橙色=未找到，红色=错误）

---

## 数据流

从摄像头到识别的完整流程:

```
摄像头帧 (640x480)
    ↓
[CameraManager] 线程安全的帧缓冲区
    ↓
[RetinaFace] 人脸检测（320x320 输入）
    ↓
对齐人脸 (112x112)
    ↓
[MobileFaceNet] 特征提取
    ↓
特征向量 (512 个浮点数)
    ↓
[FaceDatabase] 余弦相似度匹配
    ↓
识别结果（ID 或 -1）
```

---

## 构建系统

### CMake 配置

项目使用 CMake 并支持自动部署打包。

**CMake 关键特性:**
- 自动查找 Qt5、OpenCV、SQLite3
- 配置 RPATH 为 `$ORIGIN/lib` 实现便携部署
- 自动复制模型文件和库到 `deploy/` 文件夹

### 交叉编译

针对 RK3568 进行交叉编译:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/qt5/gcc_arm64 ..
make
```

### 部署包结构

执行 `make` 后，`build/deploy/` 文件夹包含:

```
deploy/
├── RK3568_FaceApp          # 可执行文件
├── lib/                    # 运行时库
│   ├── librknnrt.so
│   └── librga.so
└── assets/                 # 模型文件
    └── model/
        ├── retinaface_320.rknn
        └── w600k_mbf.rknn
```

**数据库文件** (`face_database.db`) 在首次运行时自动创建于应用工作目录。

---

## 依赖项

### 所需软件包

- **Qt5**: Widgets, Core, Gui, Multimedia, Sql
- **OpenCV**: 图像处理和摄像头 I/O
- **SQLite3**: 数据库后端
- **RKNN Runtime**: Rockchip NPU 推理引擎（位于 `3rdparty/rknn/`）
- **RGA Library**: 硬件加速图像处理（位于 `3rdparty/rga/`）

### 模型文件

- `retinaface_320.rknn`: 人脸检测模型（RetinaFace 转换为 RKNN 格式）
- `w600k_mbf.rknn`: 人脸识别模型（在 WebFace600K 数据集上训练的 MobileFaceNet）

---

## 开发指南

### 添加新功能

1. **设备层**: 修改 `CameraManager` 以支持新的传感器/输入
2. **算法层**: 参照 RetinaFace/MobileFaceNet 模式添加新 AI 模型
3. **数据库层**: 扩展 `FaceDatabase` 添加额外元数据（姓名、时间戳等）
4. **UI 层**: 更新 `MainWindow` 添加新的用户交互

### 线程安全注意事项

- **摄像头访问**: 始终使用 `getLatestFrame()` 并确保互斥锁保护
- **UI 更新**: 使用 Qt 信号/槽机制进行跨线程通信
- **模型推理**: RKNN 上下文非线程安全，如需多线程访问需序列化

### 调试技巧

- **查看日志**: 使用 `qDebug()` 语句 - 所有关键操作都会记录到控制台
- **模型加载**: 检查构造函数调用中的模型路径
- **数据库问题**: 检查 `face_database.db` 文件权限
- **摄像头问题**: 确认正确的 `/dev/videoX` 设备 ID（默认为 9）

---

## 重要实现细节

### RKNN 模型管理

- 所有 RKNN 上下文必须在构造函数中正确初始化
- 在析构函数中释放上下文以防止内存泄漏
- 模型在应用启动时加载一次

### 线程安全

- CameraManager 在独立 QThread 中运行捕获循环
- QMutex 在读写时保护 `m_latestFrame` 缓冲区
- Qt 信号确保从摄像头线程到 UI 的线程安全更新

### 内存管理

- 使用 Qt 父子对象机制实现自动清理
- RKNN 缓冲区在 init/析构函数中分配/释放
- SQLite 预处理语句使用后立即 finalize

### 硬件加速

- **NPU (RKNN)**: 处理神经网络推理（RetinaFace + MobileFaceNet）
- **RGA**: 加速图像格式转换和缩放
- **零拷贝**: 尽可能直接访问帧缓冲区

---

## 常见问题与解决方案

### 摄像头打不开

- **检查设备**: 使用 `ls /dev/video*` 查找正确的设备编号
- **权限问题**: 用户必须在 `video` 用户组中
- **修复方法**: 编辑 mainwindow.cpp 中的 `openCamera(9)` 改为正确的设备 ID

### 模型加载失败

- **检查路径**: 模型必须位于可执行文件相对路径 `assets/model/` 下
- **验证文件**: 确认 `.rknn` 文件存在且未损坏
- **RKNN 版本**: 确保 RKNN 运行时版本与模型转换版本匹配

### 数据库错误

- **权限问题**: 检查应用目录的写权限
- **重置数据库**: 删除 `face_database.db` 重新开始

### 识别准确率低

- **光照条件**: 确保人脸有良好均匀的光照
- **距离问题**: 人脸应占据画面合理比例
- **图像质量**: 避免运动模糊、部分遮挡
- **调整阈值**: 修改 FaceDatabase.h 中的 `SIMILARITY_THRESHOLD`（默认 0.6）

---

## 文件引用

讨论代码时使用行号引用:

- 人脸检测: `src/algo/RetinaFace.cpp:145` (getAlignedFaceFromCamera)
- 特征提取: `src/algo/MobileFaceNet.cpp:78` (extractFeature)
- 数据库录入: `src/db/FaceDatabase.cpp:99` (enrollFace)
- UI 录入处理: `src/ui/mainwindow.cpp:170` (on_btnEntry_clicked)
- 余弦相似度: `src/db/FaceDatabase.cpp:259` (calculateSimilarity)
