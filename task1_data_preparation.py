import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载ratings.csv文件
print("1. 加载ratings.csv文件...")
ratings_df = pd.read_csv('ratings.csv')
print(f"原始评分数据总量: {len(ratings_df)}条")

# 2. 筛选出最活跃的300个用户和300本图书（放宽限制）
print("\n2. 筛选最活跃的300个用户和300本图书...")
# 计算每个用户评价的书籍数量
user_counts = ratings_df['user_id'].value_counts()
top_users = user_counts.head(300).index.tolist()

# 计算每本书被评价的次数
book_counts = ratings_df['book_id'].value_counts()
top_books = book_counts.head(300).index.tolist()

print(f"选择了前300名活跃用户，评价书籍数量范围: {user_counts.values[299]}-{user_counts.values[0]}")
print(f"选择了前300本热门书籍，评价人数范围: {book_counts.values[299]}-{book_counts.values[0]}")

# 3. 过滤数据集，只保留这些用户对这些书的评分
print("\n3. 过滤数据集...")
filtered_df = ratings_df[
    (ratings_df['user_id'].isin(top_users)) & 
    (ratings_df['book_id'].isin(top_books))
]

# 4. 计算过滤后的评分数量
print("\n4. 计算过滤后的评分数量...")
print(f"过滤后的评分数量: {len(filtered_df)}条")

# 计算用户分布情况
user_book_counts = filtered_df['user_id'].value_counts()
print(f"过滤后数据中评价超过5本书的用户数: {sum(user_book_counts >= 5)}")
print(f"过滤后数据中每位用户平均评价书籍数: {user_book_counts.mean():.2f}")

# 5. 计算用户-项目矩阵的稀疏度（缺失项比例）
print("\n5. 计算用户-项目矩阵的稀疏度...")
# 将评分数据转换为用户-项目矩阵
user_item_matrix = filtered_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
print(f"用户-项目矩阵形状: {user_item_matrix.shape}")

# 计算稀疏度
total_elements = user_item_matrix.shape[0] * user_item_matrix.shape[1]
non_zero_elements = (user_item_matrix != 0).sum().sum()
zero_elements = total_elements - non_zero_elements
sparsity = zero_elements / total_elements

print(f"矩阵中的元素总数: {total_elements}")
print(f"非零元素数量: {non_zero_elements}")
print(f"零元素数量: {zero_elements}")
print(f"稀疏度 (缺失项比例): {sparsity:.4f} ({sparsity*100:.2f}%)")

# 6. 分析稀疏度对协同过滤的影响
print("\n6. 稀疏度对协同过滤的影响分析...")
print("高稀疏度的影响:")
print("(1) 推荐质量：高稀疏度导致难以找到足够多的相似用户或项目，降低推荐质量")
print("(2) 冷启动问题：新用户或冷门项目由于数据少，更难获得准确推荐")
print("(3) 计算效率：虽然稀疏矩阵可以节省存储空间，但可能需要特殊处理以保持计算效率")
print("(4) 可靠性：当用户评分数据不足时，基于该数据计算的相似度可能不可靠")
print("(5) 解决方案：可以采用矩阵分解、聚类或深度学习等方法处理高稀疏度数据")

# 保存过滤后的数据，供后续任务使用
filtered_df.to_csv('filtered_ratings.csv', index=False)
print("\n已将过滤后的数据保存到 'filtered_ratings.csv'，供后续任务使用")

# 可视化用户评分分布
plt.figure(figsize=(10, 6))
plt.hist(user_book_counts.values, bins=30)
plt.xlabel('用户评价的书籍数量')
plt.ylabel('用户数')
plt.title('用户评分活跃度分布')
plt.grid(True, alpha=0.3)
plt.savefig('user_ratings_distribution.png')
print("已保存用户评分分布图到 'user_ratings_distribution.png'") 