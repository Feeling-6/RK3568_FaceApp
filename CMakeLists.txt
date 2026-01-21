cmake_minimum_required(VERSION 3.14)

project(RK3568_FaceApp)

# =============================================================
# 1. 基础配置
# =============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启 Qt 的自动化处理 (必选，否则 slot/signal 不工作)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# =============================================================
# 2. 第三方库路径定义 (基于我们约定的 structure)
# =============================================================
# 这里的路径对应你整理好的:
#   project_root/3rdparty/rknn/include
#   project_root/3rdparty/rknn/lib
# -------------------------------------------------------------
set(RKNN_ROOT "${CMAKE_SOURCE_DIR}/3rdparty/rknn")
set(RGA_ROOT  "${CMAKE_SOURCE_DIR}/3rdparty/rga")

# =============================================================
# 3. 查找依赖包
# =============================================================

# --- A. 查找 Qt5 ---
# 如果在板端或交叉编译环境报错找不到 Qt，可能需要设置 CMAKE_PREFIX_PATH
find_package(Qt5 REQUIRED COMPONENTS
    Widgets
    Core
    Gui
)

# --- B. 查找 OpenCV ---
# RK3568 开发通常需要 OpenCV 做一些基础处理 (尽管核心缩放我们用 RGA)
find_package(OpenCV REQUIRED)

# --- C. 查找SQLite3包 ---
find_package(SQLite3 REQUIRED)

# =============================================================
# 4. 头文件包含路径
# =============================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/src           # 让代码里能引用到自己的头文件
    ${RKNN_ROOT}/include              # 让代码能找到 "rknn_api.h"
    ${RGA_ROOT}/include               # 让代码能找到 "RgaApi.h" / "im2d.h"
    ${OpenCV_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
)

# =============================================================
# 5. 源文件管理
# =============================================================
# 递归查找 src 目录下所有的 .cpp, .h, .ui 文件
file(GLOB_RECURSE SRC_FILES
    "src/*.cpp"
    "src/*.h"
    "src/*.ui"
)

# =============================================================
# 6. 生成可执行文件
# =============================================================
add_executable(${PROJECT_NAME} ${SRC_FILES})

# =============================================================
# 7. 链接库文件 (核心部分)
# =============================================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    # --- Qt 库 ---
    Qt5::Widgets
    Qt5::Core
    Qt5::Gui

    # --- OpenCV ---
    ${OpenCV_LIBS}

    # --- SQLite ---
    ${SQLite3_LIBRARIES}

    # --- RK3568 硬件库 (直接链接 .so 绝对路径) ---
    ${RKNN_ROOT}/lib/librknnrt.so
    ${RGA_ROOT}/lib/librga.so

    # --- 系统库 ---
    pthread
    dl
)

# =============================================================
# 8. 部署配置 (RPATH - 解决找不到库的关键)
# =============================================================
# 这一步告诉 Linux：运行这个程序时，先去它旁边的 lib/ 目录找 .so
# $ORIGIN 代表可执行文件所在的当前路径
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/lib"
)

# =============================================================
# 9. [自动化] 编译后自动整理发布包
# =============================================================
# 每次编译成功后，会自动在 build 目录下生成一个 "deploy" 文件夹
# 里面包含了你需要拷贝到板子上的所有东西
set(DEPLOY_DIR "${CMAKE_BINARY_DIR}/deploy")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    # 1. 准备目录
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEPLOY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEPLOY_DIR}/lib

    # 2. 拷贝可执行文件
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${DEPLOY_DIR}/

    # 3. 拷贝 .so 库 (从 3rdparty 拷到 deploy/lib)
    COMMAND ${CMAKE_COMMAND} -E copy ${RKNN_ROOT}/lib/librknnrt.so ${DEPLOY_DIR}/lib/
    COMMAND ${CMAKE_COMMAND} -E copy ${RGA_ROOT}/lib/librga.so ${DEPLOY_DIR}/lib/

    # 4. 拷贝模型文件 (假设你把模型放在了源码根目录的 assets/models 下)
    # 如果源目录不存在，这行可能会报 warning，请确保你放了模型
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${DEPLOY_DIR}/assets

    COMMENT "\n[部署完成] 请将 ${DEPLOY_DIR} 文件夹完整拷贝到 RK3568 板子上运行。\n"
)
