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

    // 录入人脸特征：如果已存在相同特征返回-1并设置错误信息"请不要重复录入"，成功返回录入的序号
    int enrollFace(const std::vector<float>& feature, std::string& message);

    // 识别人脸特征：找到相同特征返回序号并设置message为"你是X号"，未找到返回-1并设置message为"请先录入人脸"
    int recognizeFace(const std::vector<float>& feature, std::string& message);

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
