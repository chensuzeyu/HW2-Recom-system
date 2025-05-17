import pandas as pd

# 加载数据
ratings_df = pd.read_csv('ratings.csv')
books_df = pd.read_csv('books.csv')

# 显示ratings数据集的前几行
print("===== Ratings 数据集结构 =====")
print(f"形状: {ratings_df.shape}")
print("列名:", ratings_df.columns.tolist())
print("前5行:")
print(ratings_df.head())
print("\n")

# 显示books数据集的前几行
print("===== Books 数据集结构 =====")
print(f"形状: {books_df.shape}")
print("列名:", books_df.columns.tolist())
print("前5行:")
print(books_df.head()) 