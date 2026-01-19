/**
 * @file FaceDatabase.cpp
 * @brief 人脸特征数据库实现
 *
 * 使用SQLite存储人脸特征向量，通过余弦相似度算法实现人脸录入和识别功能。
 * 主要功能包括：防重复录入、相似度匹配识别、数据库管理等。
 */

#include "FaceDatabase.h"
#include <cmath>
#include <cstring>
#include <iostream>

/**
 * @brief 构造函数
 * 初始化数据库指针为空，标记为未初始化状态
 */
FaceDatabase::FaceDatabase() : db(nullptr), is_init(false) {
}

/**
 * @brief 析构函数
 * 释放数据库资源，关闭SQLite连接
 */
FaceDatabase::~FaceDatabase() {
    if (db) {
        sqlite3_close(db);  // 关闭数据库连接
        db = nullptr;
    }
}

/**
 * @brief 初始化数据库
 * @param db_path 数据库文件路径
 * @return 成功返回0，失败返回-1
 *
 * 打开SQLite数据库文件，如果不存在则创建，并创建人脸特征表结构
 */
int FaceDatabase::init(const std::string& db_path) {
    // 避免重复初始化
    if (is_init) {
        return 0;
    }

    // 打开或创建数据库文件
    int ret = sqlite3_open(db_path.c_str(), &db);
    if (ret != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db) << std::endl;
        return -1;
    }

    // 创建数据表
    ret = createTable();
    if (ret != 0) {
        return -1;
    }

    is_init = true;
    return 0;
}

/**
 * @brief 创建人脸特征表
 * @return 成功返回0，失败返回-1
 *
 * 表结构：
 *   - id: INTEGER 自增主键，唯一标识每个人脸
 *   - feature: BLOB 二进制数据，存储人脸特征向量(float数组)
 */
int FaceDatabase::createTable() {
    const char* sql =
        "CREATE TABLE IF NOT EXISTS faces ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"  // 自增主键
        "feature BLOB NOT NULL);";                // 特征向量的二进制存储

    char* err_msg = nullptr;
    int ret = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);

    if (ret != SQLITE_OK) {
        std::cerr << "Failed to create table: " << err_msg << std::endl;
        sqlite3_free(err_msg);  // 释放错误信息内存
        return -1;
    }

    return 0;
}

/**
 * @brief 录入人脸特征
 * @param feature 人脸特征向量(由MobileFaceNet提取的embedding)
 * @param message 输出消息，成功时为"录入成功，序号: X"，失败时为错误信息
 * @return 成功返回录入的人脸ID，失败返回-1
 *
 * 录入流程：
 *   1. 检查是否已存在相似人脸(相似度 >= 0.6)
 *   2. 如果存在则拒绝录入，提示"请不要重复录入"
 *   3. 否则将特征向量转换为BLOB格式并插入数据库
 */
int FaceDatabase::enrollFace(const std::vector<float>& feature, std::string& message) {
    if (!is_init || !db) {
        message = "数据库未初始化";
        return -1;
    }

    if (feature.empty()) {
        message = "特征向量为空";
        return -1;
    }

    // 查找是否已存在相似的人脸(防止重复录入)
    float max_similarity = 0.0f;
    int similar_id = findMostSimilar(feature, max_similarity);

    // 相似度超过阈值则认为是同一个人，拒绝录入
    if (similar_id > 0 && max_similarity >= SIMILARITY_THRESHOLD) {
        message = "请不要重复录入";
        return -1;
    }

    // 插入新的人脸特征
    const char* sql = "INSERT INTO faces (feature) VALUES (?);";
    sqlite3_stmt* stmt = nullptr;

    // 预处理SQL语句(prepared statement可防止SQL注入)
    int ret = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (ret != SQLITE_OK) {
        message = "准备SQL语句失败";
        return -1;
    }

    // 将特征向量(vector<float>)转换为BLOB二进制数据
    std::vector<unsigned char> blob = featureToBlob(feature);
    // 绑定BLOB参数到SQL语句的第1个占位符(?)
    // SQLITE_TRANSIENT表示SQLite需要复制数据
    ret = sqlite3_bind_blob(stmt, 1, blob.data(), blob.size(), SQLITE_TRANSIENT);

    if (ret != SQLITE_OK) {
        message = "绑定参数失败";
        sqlite3_finalize(stmt);  // 释放语句资源
        return -1;
    }

    // 执行SQL语句
    ret = sqlite3_step(stmt);
    if (ret != SQLITE_DONE) {
        message = "插入数据失败";
        sqlite3_finalize(stmt);
        return -1;
    }

    // 获取新插入记录的ID
    int new_id = sqlite3_last_insert_rowid(db);
    sqlite3_finalize(stmt);  // 释放语句资源

    message = "录入成功，序号: " + std::to_string(new_id);
    return new_id;
}

/**
 * @brief 识别人脸特征
 * @param feature 待识别的人脸特征向量
 * @param message 输出消息，成功时为"你是X号"，失败时为"请先录入人脸"
 * @return 识别成功返回匹配的人脸ID，失败返回-1
 *
 * 识别流程：
 *   1. 遍历数据库中所有人脸特征
 *   2. 计算与当前特征的相似度，找到最相似的人脸
 *   3. 如果相似度 >= 0.6，则认为匹配成功
 */
int FaceDatabase::recognizeFace(const std::vector<float>& feature, std::string& message) {
    if (!is_init || !db) {
        message = "数据库未初始化";
        return -1;
    }

    if (feature.empty()) {
        message = "特征向量为空";
        return -1;
    }

    // 查找最相似的人脸
    float max_similarity = 0.0f;
    int similar_id = findMostSimilar(feature, max_similarity);

    // 相似度达到阈值则认为识别成功
    if (similar_id > 0 && max_similarity >= SIMILARITY_THRESHOLD) {
        message = "你是" + std::to_string(similar_id) + "号";
        return similar_id;
    } else {
        message = "请先录入人脸";
        return -1;
    }
}

/**
 * @brief 清空人脸数据库
 *
 * 删除faces表中的所有记录，并重置自增ID计数器
 */
void FaceDatabase::clearAll() {
    if (!is_init || !db) {
        return;
    }

    // 删除所有人脸记录
    const char* sql = "DELETE FROM faces;";
    char* err_msg = nullptr;
    sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);

    if (err_msg) {
        std::cerr << "Failed to clear database: " << err_msg << std::endl;
        sqlite3_free(err_msg);
    }

    // 重置自增ID计数器(sqlite_sequence表维护自增值)
    sql = "DELETE FROM sqlite_sequence WHERE name='faces';";
    sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
}

/**
 * @brief 获取数据库中的人脸数量
 * @return 人脸数量，失败返回0
 */
int FaceDatabase::getFaceCount() {
    if (!is_init || !db) {
        return 0;
    }

    const char* sql = "SELECT COUNT(*) FROM faces;";
    sqlite3_stmt* stmt = nullptr;

    int ret = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (ret != SQLITE_OK) {
        return 0;
    }

    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);  // 获取第0列的整数值
    }

    sqlite3_finalize(stmt);
    return count;
}

/**
 * @brief 计算两个特征向量的余弦相似度
 * @param feat1 特征向量1
 * @param feat2 特征向量2
 * @return 余弦相似度值(范围0-1)，值越大表示越相似
 *
 * 余弦相似度公式：cos(θ) = (A·B) / (|A|×|B|)
 *   - A·B: 向量点积(dot product)
 *   - |A|: 向量A的欧几里得范数(L2 norm)
 *   - |B|: 向量B的欧几里得范数
 *
 * 余弦相似度表示两个向量夹角的余弦值，用于衡量方向相似性
 */
float FaceDatabase::calculateSimilarity(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    // 检查向量维度是否一致
    if (feat1.size() != feat2.size() || feat1.empty()) {
        return 0.0f;
    }

    // 计算余弦相似度
    float dot_product = 0.0f;  // 点积：Σ(A[i] * B[i])
    float norm1 = 0.0f;        // 向量1的平方和
    float norm2 = 0.0f;        // 向量2的平方和

    for (size_t i = 0; i < feat1.size(); i++) {
        dot_product += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }

    // 计算范数(norm = √(平方和))
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    // 避免除零错误
    if (norm1 < 1e-6 || norm2 < 1e-6) {
        return 0.0f;
    }

    // 返回余弦相似度 = 点积 / (范数1 × 范数2)
    return dot_product / (norm1 * norm2);
}

/**
 * @brief 在数据库中查找与给定特征最相似的人脸
 * @param feature 待匹配的特征向量
 * @param max_similarity 输出参数，返回找到的最大相似度值
 * @return 最相似人脸的ID，未找到返回-1
 *
 * 遍历数据库中所有人脸特征，计算与输入特征的相似度，返回相似度最高的人脸ID
 */
int FaceDatabase::findMostSimilar(const std::vector<float>& feature, float& max_similarity) {
    if (!is_init || !db) {
        return -1;
    }

    // 查询所有人脸记录
    const char* sql = "SELECT id, feature FROM faces;";
    sqlite3_stmt* stmt = nullptr;

    int ret = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (ret != SQLITE_OK) {
        return -1;
    }

    int best_id = -1;
    max_similarity = 0.0f;

    // 遍历所有记录
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);  // 获取人脸ID

        // 获取BLOB数据(特征向量的二进制形式)
        const unsigned char* blob = static_cast<const unsigned char*>(sqlite3_column_blob(stmt, 1));
        int blob_size = sqlite3_column_bytes(stmt, 1);

        if (blob && blob_size > 0) {
            // 将BLOB数据还原为float向量
            std::vector<float> db_feature = blobToFeature(blob, blob_size);

            // 计算相似度
            float similarity = calculateSimilarity(feature, db_feature);

            // 更新最佳匹配
            if (similarity > max_similarity) {
                max_similarity = similarity;
                best_id = id;
            }
        }
    }

    sqlite3_finalize(stmt);
    return best_id;
}

/**
 * @brief 将float向量转换为BLOB二进制数据
 * @param feature float类型的特征向量
 * @return 转换后的字节数组(unsigned char)
 *
 * 直接将float数组的内存内容拷贝为字节数组，用于SQLite的BLOB存储
 */
std::vector<unsigned char> FaceDatabase::featureToBlob(const std::vector<float>& feature) {
    // 分配足够的字节空间(每个float占4字节)
    std::vector<unsigned char> blob(feature.size() * sizeof(float));

    // 内存拷贝：将float数组按字节复制到unsigned char数组
    std::memcpy(blob.data(), feature.data(), blob.size());

    return blob;
}

/**
 * @brief 将BLOB二进制数据转换为float向量
 * @param blob 字节数组指针
 * @param size 字节数组的大小(字节数)
 * @return 还原后的float向量
 *
 * 从数据库读取的BLOB数据转换回float数组
 */
std::vector<float> FaceDatabase::blobToFeature(const unsigned char* blob, int size) {
    // 计算float元素个数(总字节数 / 每个float的字节数)
    int float_count = size / sizeof(float);

    std::vector<float> feature(float_count);

    // 内存拷贝：将字节数组按字节复制到float数组
    std::memcpy(feature.data(), blob, size);

    return feature;
}
