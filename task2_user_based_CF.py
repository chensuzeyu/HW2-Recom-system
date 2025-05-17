import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载过滤后的数据和图书信息
print("1. 加载数据...")
ratings_df = pd.read_csv('filtered_ratings.csv')
books_df = pd.read_csv('books.csv')

# 2. 构建用户-项目矩阵
print("\n2. 构建用户-项目矩阵...")
user_item_matrix = ratings_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
print(f"用户-项目矩阵形状: {user_item_matrix.shape}")

# 3. 计算用户间的余弦相似度
print("\n3. 计算用户间的余弦相似度...")
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, 
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)
print(f"用户相似度矩阵形状: {user_similarity_df.shape}")

# 4. 选择评分最多的用户作为目标用户（不再要求至少评价5本书）
print("\n4. 选择评分最多的用户...")
user_ratings_count = ratings_df['user_id'].value_counts()

# 选择评分数量最多的用户
target_user_id = user_ratings_count.index[0]
print(f"已选择用户ID: {target_user_id}，该用户评价过 {user_ratings_count[target_user_id]} 本书")

# 5. 基于用户相似度推荐书籍
print("\n5. 基于用户相似度推荐书籍...")

# 获取目标用户已评价和未评价的书籍
user_rated_books = ratings_df[ratings_df['user_id'] == target_user_id]['book_id'].unique()
all_books = ratings_df['book_id'].unique()
user_unrated_books = np.setdiff1d(all_books, user_rated_books)

print(f"用户已评价书籍数量: {len(user_rated_books)}")
print(f"用户未评价书籍数量: {len(user_unrated_books)}")

# 获取目标用户的相似用户（排除自己）
user_similarities = user_similarity_df[target_user_id].drop(target_user_id)
most_similar_users = user_similarities.sort_values(ascending=False).head(10).index

# 计算未评价书籍的预测评分
predictions = {}

for book_id in user_unrated_books:
    book_ratings = []
    weights = []
    
    for user_id in most_similar_users:
        rating = user_item_matrix.loc[user_id, book_id]
        if rating > 0:  # 只考虑实际评分，忽略缺失值（已填充为0）
            similarity = user_similarity_df.loc[target_user_id, user_id]
            # 确保只有当相似度为正值时才添加评分和权重
            if similarity > 0:
                book_ratings.append(rating)
                weights.append(similarity)
    
    # 如果有足够的相似用户评价过这本书，计算加权平均分
    if len(book_ratings) > 0 and sum(weights) > 0:
        try:
            predictions[book_id] = np.average(book_ratings, weights=weights)
        except ZeroDivisionError:
            # 如果出现除零错误，使用简单平均
            predictions[book_id] = np.mean(book_ratings)

# 按预测评分排序，推荐前5本书（或可用的最大数量）
max_recommendations = min(5, len(predictions))
if max_recommendations == 0:
    print("\n未能找到可推荐的书籍。这可能是因为数据稀疏导致没有找到相似用户评价的书籍。")
else:
    recommended_books = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:max_recommendations]

    print(f"\n推荐给用户 {target_user_id} 的 {max_recommendations} 本书籍:")
    print("\n{:<10} {:<50} {:<20} {:<10}".format('Book ID', 'Title', 'Author', 'Predicted Rating'))
    print("-" * 90)

    for book_id, predicted_rating in recommended_books:
        # 确保book_id存在于books_df中
        book_info_df = books_df[books_df['book_id'] == book_id]
        if not book_info_df.empty:
            book_info = book_info_df.iloc[0]
            print("{:<10} {:<50} {:<20} {:<10.2f}".format(
                book_id, 
                book_info['title'][:50], 
                book_info['authors'][:20], 
                predicted_rating
            ))
        else:
            print(f"{book_id:<10} {'未找到书籍信息':<50} {'未知':<20} {predicted_rating:<10.2f}")

# 6. 分析选择评分较多用户对推荐质量的影响
print("\n6. 分析选择评分较多用户对推荐质量的影响:")
print("(1) 数据稀疏挑战：当前数据集非常稀疏，很少有用户评价了多本书，这降低了推荐质量")
print("(2) 用户偏好表达：评分较多的用户偏好特征更明确，系统能更准确理解其品味")
print("(3) 相似性计算：评分数据越丰富，用户相似度计算越可靠")
print("(4) 稀疏矩阵的局限：在极度稀疏的矩阵中，即使选择评分最多的用户，相似度计算仍可能不可靠")
print("(5) 可能解决方案：可考虑使用内容特征增强推荐、应用降维技术或引入隐式反馈数据") 