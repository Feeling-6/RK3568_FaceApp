# README.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RK3568-based face recognition application featuring real-time face detection, enrollment, and recognition. Built with Qt5, OpenCV, and Rockchip's RKNN runtime for NPU hardware acceleration.

### Key Features

- **Real-time Face Detection**: Detects faces from camera feed using RetinaFace model
- **Face Enrollment**: Captures and stores face features in SQLite database
- **Face Recognition**: Matches detected faces against stored database using cosine similarity
- **Hardware Acceleration**: Leverages RK3568 NPU via RKNN runtime for fast inference
- **Thread-safe Design**: Multi-threaded architecture for smooth UI and real-time processing
- **Simple Deployment**: Single folder deployment with all dependencies included

---

## Quick Start

### Build the Project

```bash
mkdir -p build && cd build
cmake ..
make
```

### Deploy to RK3568 Board

After building, the `build/deploy/` folder contains everything needed:

```bash
# Copy the entire deploy folder to the board
scp -r deploy/ user@rk3568:/path/to/app/

# On the board, run the application
cd /path/to/app/deploy
./RK3568_FaceApp
```

### Usage

1. **Launch Application**: The app opens in fullscreen with camera preview
2. **Enroll Face**: Click "录入" (Entry) button to add a new face to database
3. **Recognize Face**: Click "识别" (Recognize) button to identify the person

---

## Project Structure

```
RK3568_FaceApp/
├── assets/
│   └── model/              # RKNN model files
│       ├── retinaface_320.rknn
│       └── w600k_mbf.rknn
├── src/
│   ├── main.cpp            # Application entry point
│   ├── device/             # Device layer (camera)
│   │   ├── CameraManager.h
│   │   └── CameraManager.cpp
│   ├── algo/               # Algorithm layer (AI models)
│   │   ├── RetinaFace.h/cpp
│   │   └── MobileFaceNet.h/cpp
│   ├── db/                 # Database layer
│   │   ├── FaceDatabase.h
│   │   └── FaceDatabase.cpp
│   └── ui/                 # UI layer
│       ├── mainwindow.h/cpp
│       └── mainwindow.ui
├── 3rdparty/               # Third-party libraries
│   ├── rknn/               # RKNN runtime headers and libs
│   └── rga/                # RGA image processing libs
└── CMakeLists.txt
```

---

## Architecture

### Four-Layer Architecture

The application follows a clean four-layer architecture for modularity and maintainability:

```
┌─────────────────────────────────────────────────┐
│              UI Layer (Qt GUI)                  │
│  - MainWindow: Display & User Interaction       │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│           Database Layer (SQLite)               │
│  - FaceDatabase: Store & Match Face Features    │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│         Algorithm Layer (RKNN Models)           │
│  - RetinaFace: Face Detection & Alignment       │
│  - MobileFaceNet: Feature Extraction            │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│          Device Layer (Camera I/O)              │
│  - CameraManager: Thread-safe Frame Capture     │
└─────────────────────────────────────────────────┘
```

### 1. Device Layer (`src/device/`)

**CameraManager** - QThread-based camera capture

- Opens `/dev/videoX` devices using OpenCV VideoCapture
- Runs continuous frame capture in background thread
- Emits Qt signals (`newFrameCaptured`) for UI updates
- Provides thread-safe `getLatestFrame()` for algorithm access
- Uses QMutex to protect frame buffer

**Key Methods:**
- `openCamera(int deviceId)`: Open camera device
- `getLatestFrame(cv::Mat& frame)`: Get current frame (thread-safe)

### 2. Algorithm Layer (`src/algo/`)

#### RetinaFace - Face Detection

- **Input**: 320x320 RGB images from camera
- **Output**: Face bounding boxes, confidence scores, 5 facial landmarks
- **Acceleration**: RKNN (NPU inference)

**Key Methods:**
- `init()`: Load RKNN model and initialize
- `getAlignedFaceFromCamera(CameraManager& camera)`: Detect face and return aligned 112x112 image

**Implementation Details:**
- Pre-computed anchors (BOX_PRIORS_320) in `initPriors()`
- Face alignment using similarity transformation based on 5 landmarks
- Returns aligned and cropped 112x112 face image ready for feature extraction

#### MobileFaceNet - Feature Extraction

- **Input**: 112x112 aligned face images
- **Output**: 512-dimensional feature vectors (embeddings)
- **Acceleration**: RKNN (NPU inference)

**Key Methods:**
- `init(const std::string& modelPath)`: Load RKNN model
- `extractFeature(const cv::Mat& alignedFace, std::vector<float>& feature)`: Extract face embedding

### 3. Database Layer (`src/db/`)

**FaceDatabase** - SQLite-based face feature storage

- **Storage**: Face features stored as BLOB (binary large object)
- **Matching**: Cosine similarity algorithm for face recognition
- **Threshold**: 0.6 similarity score for positive match

**Database Schema:**
```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature BLOB NOT NULL
);
```

**Key Methods:**
- `init(const std::string& db_path)`: Initialize SQLite database
- `enrollFace(const std::vector<float>& feature, std::string& message)`: Add new face (with duplicate check)
- `recognizeFace(const std::vector<float>& feature, std::string& message)`: Find matching face
- `clearAll()`: Delete all faces
- `getFaceCount()`: Get number of enrolled faces

**Recognition Algorithm:**
- Uses cosine similarity: `cos(θ) = (A·B) / (|A|×|B|)`
- Iterates all stored faces to find best match
- Returns ID if similarity >= 0.6, otherwise returns -1

### 4. UI Layer (`src/ui/`)

**MainWindow** - Qt-based graphical interface

- Displays real-time camera feed using QLabel
- Two main buttons: "录入" (Enroll) and "识别" (Recognize)
- Status label shows operation results with color coding

**UI Workflow:**

**Face Enrollment (`on_btnEntry_clicked`):**
1. Get aligned face from camera → `RetinaFace::getAlignedFaceFromCamera()`
2. Extract features → `MobileFaceNet::extractFeature()`
3. Store in database → `FaceDatabase::enrollFace()`
4. Display result (green=success, orange=duplicate, red=error)

**Face Recognition (`on_btnRecognize_clicked`):**
1. Get aligned face from camera → `RetinaFace::getAlignedFaceFromCamera()`
2. Extract features → `MobileFaceNet::extractFeature()`
3. Match in database → `FaceDatabase::recognizeFace()`
4. Display result (green=recognized "你是X号", orange=not found, red=error)

---

## Data Flow

Complete pipeline from camera to recognition:

```
Camera Frame (640x480)
    ↓
[CameraManager] Thread-safe frame buffer
    ↓
[RetinaFace] Face detection (320x320 input)
    ↓
Aligned Face (112x112)
    ↓
[MobileFaceNet] Feature extraction
    ↓
Feature Vector (512 floats)
    ↓
[FaceDatabase] Cosine similarity matching
    ↓
Recognition Result (ID or -1)
```

---

## Build System

### CMake Configuration

The project uses CMake with automatic deployment packaging.

**Key CMake Features:**
- Finds Qt5, OpenCV, SQLite3
- Configures RPATH to `$ORIGIN/lib` for portable deployment
- Auto-copies models and libraries to `deploy/` folder

### Cross-Compilation

For RK3568 cross-compilation:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/qt5/gcc_arm64 ..
make
```

### Deploy Package Structure

After `make`, the `build/deploy/` folder contains:

```
deploy/
├── RK3568_FaceApp          # Executable
├── lib/                    # Runtime libraries
│   ├── librknnrt.so
│   └── librga.so
└── assets/                 # Model files
    └── model/
        ├── retinaface_320.rknn
        └── w600k_mbf.rknn
```

**Database file** (`face_database.db`) is created automatically in the app's working directory on first run.

---

## Dependencies

### Required Packages

- **Qt5**: Widgets, Core, Gui, Multimedia, Sql
- **OpenCV**: Image processing and camera I/O
- **SQLite3**: Database backend
- **RKNN Runtime**: Rockchip NPU inference engine (in `3rdparty/rknn/`)
- **RGA Library**: Hardware-accelerated image processing (in `3rdparty/rga/`)

### Model Files

- `retinaface_320.rknn`: Face detection model (RetinaFace converted to RKNN format)
- `w600k_mbf.rknn`: Face recognition model (MobileFaceNet trained on WebFace600K)

---

## Development Guide

### Adding New Features

1. **Device Layer**: Modify `CameraManager` for new sensors/inputs
2. **Algorithm Layer**: Add new AI models by following RetinaFace/MobileFaceNet pattern
3. **Database Layer**: Extend `FaceDatabase` for additional metadata (names, timestamps, etc.)
4. **UI Layer**: Update `MainWindow` for new user interactions

### Thread Safety Considerations

- **Camera Access**: Always use `getLatestFrame()` with mutex protection
- **UI Updates**: Use Qt signals/slots for cross-thread communication
- **Model Inference**: RKNN contexts are not thread-safe; serialize access if needed

### Debugging Tips

- **Check Logs**: Use `qDebug()` statements - all critical operations log to console
- **Model Loading**: Verify model paths in constructor calls
- **Database Issues**: Check `face_database.db` file permissions
- **Camera Problems**: Confirm correct `/dev/videoX` device ID (default is 9)

---

## Important Implementation Details

### RKNN Model Management

- All RKNN contexts must be properly initialized in constructor
- Release contexts in destructor to prevent memory leaks
- Models are loaded once at application startup

### Thread Safety

- CameraManager runs capture loop in separate QThread
- QMutex protects `m_latestFrame` buffer during read/write
- Qt signals ensure thread-safe UI updates from camera thread

### Memory Management

- Uses Qt parent-child ownership for automatic cleanup
- RKNN buffers allocated/freed in init/destructor
- SQLite prepared statements finalized after use

### Hardware Acceleration

- **NPU (RKNN)**: Handles neural network inference (RetinaFace + MobileFaceNet)
- **RGA**: Accelerates image format conversion and scaling
- **Zero-copy**: Direct frame buffer access where possible

---

## Common Issues & Solutions

### Camera not opening

- **Check device**: `ls /dev/video*` to find correct device number
- **Permissions**: User must be in `video` group
- **Fix**: Edit `openCamera(9)` in mainwindow.cpp to correct device ID

### Model loading fails

- **Check paths**: Models must be in `assets/model/` relative to executable
- **Verify files**: Confirm `.rknn` files exist and are not corrupted
- **RKNN version**: Ensure RKNN runtime matches model conversion version

### Database errors

- **Permissions**: Check write permissions in app directory
- **Reset DB**: Delete `face_database.db` to start fresh

### Poor recognition accuracy

- **Lighting**: Ensure good, even lighting on face
- **Distance**: Face should fill reasonable portion of frame
- **Quality**: Avoid motion blur, partial faces
- **Threshold**: Adjust `SIMILARITY_THRESHOLD` in FaceDatabase.h (default 0.6)

---

## File References

When discussing code, use line number references:

- Face detection: `src/algo/RetinaFace.cpp:145` (getAlignedFaceFromCamera)
- Feature extraction: `src/algo/MobileFaceNet.cpp:78` (extractFeature)
- Database enrollment: `src/db/FaceDatabase.cpp:99` (enrollFace)
- UI enrollment handler: `src/ui/mainwindow.cpp:170` (on_btnEntry_clicked)
- Cosine similarity: `src/db/FaceDatabase.cpp:259` (calculateSimilarity)
