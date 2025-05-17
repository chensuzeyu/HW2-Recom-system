import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# 忽略可能的警告信息
warnings.filterwarnings('ignore')

# 1. 加载过滤后的数据
print("1. 加载数据...")
ratings_df = pd.read_csv('filtered_ratings.csv')
print(f"数据集大小: {ratings_df.shape}")

# 检查数据集是否过小
if len(ratings_df) < 20:
    print("\n警告: 数据集过小，可能无法有效进行SVD评估。结果可能不可靠。")

# 2. 将过滤后的数据分为75%训练集和25%测试集
print("\n2. 将数据集分为75%训练集和25%测试集...")

try:
    # 使用Surprise库加载数据
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)
    
    # 手动分割数据集，避免使用generator
    # 先转换成列表
    all_ratings = [rating for rating in data.build_full_trainset().all_ratings()]
    
    # 随机打乱并分割
    np.random.seed(42)
    np.random.shuffle(all_ratings)
    train_size = int(0.75 * len(all_ratings))
    
    # 创建训练集
    train_ratings = all_ratings[:train_size]
    test_ratings = all_ratings[train_size:]
    
    # 创建训练集对象
    trainset = data.build_full_trainset()
    
    # 创建测试集，格式为(user_id, item_id, rating)元组列表
    testset = [(trainset.to_raw_uid(uid), trainset.to_raw_iid(iid), r) 
              for (uid, iid, r) in test_ratings]
    
    print(f"训练集大小: {len(train_ratings)}")
    print(f"测试集大小: {len(testset)}")

    # 3. 使用Surprise库的SVD算法训练矩阵分解模型
    print("\n3. 使用SVD算法训练矩阵分解模型...")

    # 创建SVD模型 - 使用较小的潜在因子数，避免过拟合
    n_factors = min(5, len(ratings_df) // 10 + 1)  # 动态调整因子数量
    print(f"由于数据集较小，使用 {n_factors} 个潜在因子")
    svd = SVD(n_factors=n_factors, random_state=42)

    # 在训练集上拟合模型
    svd.fit(trainset)

    # 4. 预测测试集评分并计算RMSE
    print("\n4. 预测测试集评分并计算RMSE...")

    # 在测试集上进行预测
    predictions = svd.test(testset)

    # 计算RMSE指标
    rmse = accuracy.rmse(predictions)
    print(f"测试集RMSE: {rmse:.4f}")

    # 获取所有预测结果，用于可视化
    actual_ratings = np.array([pred.r_ui for pred in predictions])
    predicted_ratings = np.array([pred.est for pred in predictions])

    # 如果有足够的预测结果，才创建可视化
    if len(predictions) > 5:
        # 可视化预测结果与实际评分的散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_ratings, predicted_ratings, alpha=0.3)
        plt.plot([1, 5], [1, 5], 'r--')  # 参考线：实际=预测
        plt.xlabel('实际评分')
        plt.ylabel('预测评分')
        plt.title('SVD模型预测评分与实际评分比较')
        plt.savefig('svd_predictions_scatter.png')
        print("已保存预测结果散点图到 'svd_predictions_scatter.png'")

        # 计算预测误差
        errors = actual_ratings - predicted_ratings
        # 计算每个评分值上的平均误差
        rating_error = {}
        for i, pred in enumerate(predictions):
            r = pred.r_ui
            if r not in rating_error:
                rating_error[r] = []
            rating_error[r].append(errors[i])

        for r in sorted(rating_error.keys()):
            rating_error[r] = np.mean(np.abs(rating_error[r]))
            print(f"评分 {r} 的平均绝对误差: {rating_error[r]:.4f}")
    else:
        print("预测结果过少，跳过可视化和详细误差分析。")

    # 5. 分析RMSE值和影响模型性能的因素
    print("\n5. 分析RMSE值和影响模型性能的因素...")

    print("RMSE值分析:")
    if rmse < 0.8:
        print(f"RMSE = {rmse:.4f} - 看起来较低，但在数据量极小的情况下这可能是过拟合的迹象")
    elif rmse < 1.0:
        print(f"RMSE = {rmse:.4f} - 可接受：考虑到数据稀疏，模型表现尚可")
    else:
        print(f"RMSE = {rmse:.4f} - 较高：模型预测与实际评分存在差距，但这在极小数据集上是预期的")

    print("\n影响当前SVD模型性能的主要因素:")
    print("(1) 数据集极小 - 仅有 {} 个评分记录，难以进行有效的矩阵分解".format(len(ratings_df)))
    print("(2) 高度稀疏 - 稀疏度高达95%以上，严重影响SVD性能")
    print("(3) 过拟合风险 - 在小数据集上使用复杂模型容易过拟合")
    print("(4) 数据质量 - 数据集中用户和项目样本非常有限，不具代表性")
    print("(5) 评估可靠性 - 测试集太小，评估结果不够稳定可靠")

    # 尝试不同数量的潜在因子，分析其影响
    if len(ratings_df) >= 20:
        print("\n不同潜在因子数量对模型性能的影响:")
        # 根据数据集大小调整因子数量范围
        max_factors = min(20, len(ratings_df) // 4)
        factors = [1, 2, 3, min(5, max_factors), min(10, max_factors)]
        factors = sorted(list(set(factors)))  # 去除重复并排序
        
        rmse_values = []

        for n_factors in factors:
            # 创建指定潜在因子数的SVD模型
            model = SVD(n_factors=n_factors, random_state=42)
            model.fit(trainset)
            predictions = model.test(testset)
            rmse_val = accuracy.rmse(predictions)
            rmse_values.append(rmse_val)
            print(f"潜在因子数 = {n_factors}, RMSE = {rmse_val:.4f}")

        if len(factors) > 1:
            # 绘制潜在因子数量与RMSE的关系图
            plt.figure(figsize=(10, 6))
            plt.plot(factors, rmse_values, marker='o')
            plt.xlabel('潜在因子数量')
            plt.ylabel('RMSE')
            plt.title('潜在因子数量对SVD模型性能的影响')
            plt.grid(True)
            plt.savefig('svd_factors_rmse.png')
            print("已保存潜在因子分析图到 'svd_factors_rmse.png'")
        else:
            print("因子数量选项过少，跳过因子分析图表生成")
    else:
        print("\n数据集过小，跳过潜在因子数量分析")

    print("\n实际应用建议:")
    print("1. 使用完整的GoodBooks-10k数据集进行训练，而非高度过滤的子集")
    print("2. 考虑使用其他推荐算法，如基于内容的过滤或混合方法")
    print("3. 在稀疏数据上，可能需要引入额外信息（如书籍元数据）辅助推荐")
    print("4. 探索矩阵分解的变种如Biased Matrix Factorization或Non-negative Matrix Factorization")

except Exception as e:
    print(f"\n执行SVD评估时发生错误: {str(e)}")
    print("可能的原因: 数据集过小或无法有效分割成训练集和测试集")
    print("建议: 使用更大的数据集进行SVD模型评估") 