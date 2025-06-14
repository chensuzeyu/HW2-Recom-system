# GoodBooks图书推荐系统

## 项目概述
本项目基于GoodBooks-10k数据集实现了一个完整的图书推荐系统，包含了数据准备、基于用户的协同过滤、基于项目的协同过滤和SVD模型评估四个主要任务。项目旨在展示不同推荐算法的实现和性能对比，以及如何处理高稀疏度数据集的挑战。

## 数据集说明
- **数据源**：GoodBooks-10k数据集
- **主要文件**：
  - `ratings.csv`：包含约1,000,000条用户评分数据
  - `books.csv`：包含10,000本书的详细信息（标题、作者、ISBN等）
- **数据特点**：评分范围为1-5星，数据集较大但非常稀疏

## 项目结构
```
├── Book-Recomm-Sys.ipynb     # Jupyter Notebook集成版本
├── README.md                 # 项目说明（英文）
├── README-ZH.md              # 项目说明（中文）
├── requirements.txt          # 依赖包列表
├── main.py                   # 主程序（顺序执行所有任务）
├── inspect_data.py           # 数据集结构初步分析
├── task1_data_preparation.py # 数据准备与稀疏度分析
├── task2_user_based_CF.py    # 基于用户的协同过滤算法
├── task3_item_based_CF.py    # 基于项目的协同过滤算法
├── task4_SVD_evaluation.py   # SVD矩阵分解模型评估
├── books.csv                 # 书籍信息数据
├── ratings.csv               # 用户评分数据
├── filtered_ratings.csv      # 过滤后的评分数据
├── user_ratings_distribution.png  # 用户评分分布图
├── svd_predictions_scatter.png    # SVD预测结果散点图
└── svd_factors_rmse.png           # 潜在因子分析图
```
 
## 功能模块详解

### 1. 数据准备与稀疏度分析
- **功能**：筛选活跃用户和热门书籍，分析稀疏度
- **实现**：
  - 筛选最活跃的300个用户和300本热门图书
  - 计算并分析用户-项目矩阵的稀疏度
  - 生成用户评分分布可视化
- **输出**：`filtered_ratings.csv`、`user_ratings_distribution.png`
- **核心指标**：矩阵稀疏度（通常>95%）

### 2. 基于用户的协同过滤
- **功能**：根据用户相似度为目标用户推荐书籍
- **实现**：
  - 构建用户-项目评分矩阵
  - 计算用户间余弦相似度
  - 基于相似用户的评分加权平均计算预测评分
  - 推荐评分最高的未读书籍
- **优势**：直观易理解，能发现用户潜在兴趣
- **劣势**：在高稀疏度数据上性能下降，计算开销大

### 3. 基于项目的协同过滤
- **功能**：根据项目相似度推荐与用户喜欢的书相似的书籍
- **实现**：
  - 构建项目-项目相似度矩阵
  - 从用户高分评价的书籍出发寻找相似书籍
  - 显示推荐书籍详细信息
- **优势**：稳定性好，新用户冷启动问题较小
- **劣势**：内容多样性可能不足，难以发现完全不同类型的兴趣

### 4. SVD矩阵分解模型
- **功能**：使用矩阵分解技术降维并预测用户评分
- **实现**：
  - 使用Surprise库的SVD算法训练模型
  - 分析不同潜在因子数量对性能的影响
  - 计算预测RMSE并可视化结果
- **优势**：能有效处理稀疏数据，理论基础扎实
- **劣势**：参数调优复杂，解释性较差

## 实现中的挑战与解决方案

### 1. 数据稀疏度问题
- **挑战**：过滤后的数据集稀疏度极高(>95%)，影响推荐质量
- **解决方案**：
  - 扩大筛选范围至300个用户/书籍（原计划100个）
  - 动态调整相似度计算方法和阈值
  - 在权重为零时使用简单平均代替加权平均

### 2. 用户推荐条件问题
- **挑战**：很少有用户评价5本以上的书，难以找到合适的目标用户
- **解决方案**：
  - 取消最低评价数量阈值
  - 直接选择评价数量最多的用户作为目标用户
  - 增加异常情况处理逻辑

### 3. 计算效率与准确性平衡
- **挑战**：相似度计算开销大，且在稀疏数据上准确性不足
- **解决方案**：
  - 优化余弦相似度计算过程
  - 增加边缘情况处理
  - 为SVD模型动态调整潜在因子数量

### 4. 书籍信息匹配问题
- **挑战**：某些book_id在books.csv中不存在对应信息
- **解决方案**：
  - 添加ID存在性检查
  - 使用适当的占位信息代替缺失数据
  - 增强错误处理能力

## 系统性能与指标
- **稀疏度**：典型稀疏度在95-98%之间
- **推荐准确性**：SVD模型RMSE通常在0.8-1.2之间
- **计算效率**：在过滤后的数据集上运行速度快，全数据集需优化
- **冷启动处理**：系统能够处理新用户，但推荐质量可能不佳

## 运行方法
```bash
# 安装依赖
pip install -r requirements.txt

# 运行所有任务
python main.py

# 或单独运行各任务
python task1_data_preparation.py
python task2_user_based_CF.py
python task3_item_based_CF.py
python task4_SVD_evaluation.py
```

## 性能优化与未来改进方向
1. **数据层面**：
   - 使用完整数据集进行训练，而非高度过滤的子集
   - 考虑引入隐式反馈数据增强稀疏矩阵
   - 使用更丰富的书籍元数据辅助推荐

2. **算法层面**：
   - 实现混合推荐系统，结合内容和协同过滤
   - 尝试深度学习推荐模型如NCF、DeepFM等
   - 实现更多种类的矩阵分解技术处理稀疏数据

3. **评估层面**：
   - 增加准确率、召回率、F1值等更全面的评估指标
   - 实现交叉验证防止过拟合
   - 添加A/B测试框架评估推荐质量

4. **工程层面**：
   - 优化大规模数据处理性能
   - 增加增量更新推荐模型的能力
   - 实现API接口供外部系统调用

## 依赖环境
- Python 3.6+
- pandas
- numpy
- scikit-learn==1.0.2
- scikit-surprise==1.1.3
- matplotlib==3.5.3

## 作者
- [您的姓名]

## 许可证
- MIT License 