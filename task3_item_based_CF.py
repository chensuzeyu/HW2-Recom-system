import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载数据
print("1. 加载数据...")
ratings_df = pd.read_csv('filtered_ratings.csv')
books_df = pd.read_csv('books.csv')

# 2. 实现基于项目的协同过滤模型
print("\n2. 实现基于项目的协同过滤模型...")
# 构建用户-项目矩阵，并转置为项目-用户矩阵用于计算项目间相似度
user_item_matrix = ratings_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
item_user_matrix = user_item_matrix.T  # 转置，得到项目-用户矩阵
print(f"项目-用户矩阵形状: {item_user_matrix.shape}")

# 计算项目间的余弦相似度
print("计算项目间的余弦相似度...")
item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=item_user_matrix.index,
                                  columns=item_user_matrix.index)
print(f"项目相似度矩阵形状: {item_similarity_df.shape}")

# 3. 选择目标用户评价过的任意一本书作为参考（优先高分评价≥4）
print("\n3. 选择目标用户高分评价过的一本书...")
# 先选一个有评分的用户
user_ratings_count = ratings_df['user_id'].value_counts()
target_user_id = user_ratings_count.index[0]  # 选择评分最多的用户

# 获取该用户评价的所有书籍
user_books = ratings_df[ratings_df['user_id'] == target_user_id]

# 优先选择高分评价的书（评分≥4），如果没有，则选择任意评价过的书
high_rated_books = user_books[user_books['rating'] >= 4]

if len(high_rated_books) > 0:
    # 有高分评价的书
    target_row = high_rated_books.iloc[0]
else:
    # 无高分评价，选择任意一本
    if len(user_books) > 0:
        target_row = user_books.iloc[0]
    else:
        print("错误: 未找到目标用户评价的任何书籍!")
        exit(1)

target_book_id = target_row['book_id']
target_book_rating = target_row['rating']

# 确保target_book_id存在于books_df中
target_book_df = books_df[books_df['book_id'] == target_book_id]
if target_book_df.empty:
    print(f"警告: 书籍ID {target_book_id} 在books.csv中未找到，将使用简化信息")
    target_book_title = f"未知书籍 (ID: {target_book_id})"
    target_book_author = "未知"
else:
    target_book_info = target_book_df.iloc[0]
    target_book_title = target_book_info['title']
    target_book_author = target_book_info['authors']

print(f"已选择用户 {target_user_id} 评价的书籍:")
print(f"Book ID: {target_book_id}")
print(f"标题: {target_book_title}")
print(f"作者: {target_book_author}")
print(f"用户评分: {target_book_rating}")

# 4. 计算项目间相似度，找出5本相似书籍
print("\n4. 计算项目间相似度，找出相似书籍...")

# 检查目标书籍是否在相似度矩阵中
if target_book_id not in item_similarity_df.index:
    print(f"错误: 在相似度矩阵中未找到书籍ID {target_book_id}!")
    exit(1)

# 获取与目标书籍最相似的书籍（排除自己）
book_similarities = item_similarity_df[target_book_id].drop(target_book_id)
# 根据实际情况，可能需要调整推荐数量
max_recommendations = min(5, len(book_similarities))

if max_recommendations == 0:
    print("没有找到相似的书籍。这可能是因为数据太稀疏。")
else:
    similar_books_ids = book_similarities.sort_values(ascending=False).head(max_recommendations).index
    
    print(f"\n与《{target_book_title}》最相似的{max_recommendations}本书籍:")
    print("\n{:<10} {:<50} {:<20} {:<10}".format('Book ID', 'Title', 'Author', 'Similarity'))
    print("-" * 90)
    
    for book_id in similar_books_ids:
        # 确保book_id存在于books_df中
        book_info_df = books_df[books_df['book_id'] == book_id]
        similarity = book_similarities[book_id]
        
        if not book_info_df.empty:
            book_info = book_info_df.iloc[0]
            print("{:<10} {:<50} {:<20} {:<10.4f}".format(
                book_id,
                book_info['title'][:50],
                book_info['authors'][:20],
                similarity
            ))
        else:
            print("{:<10} {:<50} {:<20} {:<10.4f}".format(
                book_id,
                f"未知书籍 (ID: {book_id})",
                "未知",
                similarity
            ))

# 5. 对比基于用户和基于项目的协同过滤优缺点
print("\n5. 基于用户与基于项目的协同过滤比较:")
print("\n基于用户的协同过滤优缺点:")
print("优点:")
print("(1) 能够捕捉用户的多样化兴趣，推荐不同类别的项目")
print("(2) 适合用户数量少于项目数量的系统")
print("(3) 可以为新项目快速生成推荐，缓解项目冷启动问题")
print("\n缺点:")
print("(1) 在当前极度稀疏的数据集中表现不佳，难以找到足够多的相似用户")
print("(2) 用户数量增长时计算复杂度高，扩展性差")
print("(3) 用户兴趣变化快，基于历史数据的相似度可能很快过时")

print("\n基于项目的协同过滤优缺点:")
print("优点:")
print("(1) 在稀疏数据中通常比基于用户的方法表现更好")
print("(2) 项目特征通常比用户兴趣更稳定，推荐结果更一致")
print("(3) 可以预计算项目相似度，提高实时推荐效率")
print("\n缺点:")
print("(1) 在当前过滤后的极小数据集中，项目间的相似度计算仍然不够可靠")
print("(2) 推荐结果可能过度专注于某一类似项目")
print("(3) 对新用户的冷启动问题处理能力较弱") 