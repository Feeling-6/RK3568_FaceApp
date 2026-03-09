#ifndef FACEDATABASE_H
#define FACEDATABASE_H

#include <string>
#include <vector>
#include <sqlite3.h>

class FaceDatabase {
public:
    FaceDatabase();
    ~FaceDatabase();

    // 初始化数据库
    int init(const std::string& db_path);

    // 录入人脸特征：成功返回true并通过outId返回录入序号，失败返回false（如已存在相同特征）
    bool enrollFace(const std::vector<float>& feature, int& outId);

    // 识别人脸特征：成功返回true并通过outId返回匹配序号，失败返回false（未找到匹配）
    bool recognizeFace(const std::vector<float>& feature, int& outId);

    // 清空数据库
    void clearAll();

    // 获取数据库中的人脸数量
    int getFaceCount();

private:
    sqlite3* db;
    bool is_init;

    // 相似度阈值，余弦相似度大于此值认为是同一个人
    const float SIMILARITY_THRESHOLD = 0.6f;

    // 创建表
    int createTable();

    // 计算两个特征向量的余弦相似度
    float calculateSimilarity(const std::vector<float>& feat1, const std::vector<float>& feat2);

    // 在数据库中查找最相似的人脸，返回id和相似度
    int findMostSimilar(const std::vector<float>& feature, float& max_similarity);

    // 将vector<float>转换为blob数据
    std::vector<unsigned char> featureToBlob(const std::vector<float>& feature);

    // 将blob数据转换为vector<float>
    std::vector<float> blobToFeature(const unsigned char* blob, int size);
};

#endif //FACEDATABASE_H
