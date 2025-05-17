import os
import subprocess
import time

def run_task(task_file, task_name):
    print("\n" + "="*80)
    print(f"执行任务: {task_name}")
    print("="*80)
    
    try:
        # 使用subprocess运行Python脚本
        process = subprocess.Popen(['python', task_file], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        
        # 实时显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        # 等待进程结束
        process.wait()
        
        # 如果有错误，打印错误信息
        if process.returncode != 0:
            print(f"\n错误: {task_file} 执行失败")
            stderr = process.stderr.read()
            if stderr:
                print(f"错误信息:\n{stderr}")
        else:
            print(f"\n任务完成: {task_name}")
            
    except Exception as e:
        print(f"\n执行 {task_file} 时发生异常: {str(e)}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    # 跳过安装依赖包，假设已经安装好了所需的包
    print("开始执行各任务，请确保已安装所需的依赖包...")
    
    # 定义任务
    tasks = [
        ("task1_data_preparation.py", "任务1 - 数据准备与稀疏度分析"),
        ("task2_user_based_CF.py", "任务2 - 基于用户的协同过滤"),
        ("task3_item_based_CF.py", "任务3 - 基于项目的协同过滤"),
        ("task4_SVD_evaluation.py", "任务4 - SVD模型评估")
    ]
    
    # 依次执行每个任务
    for task_file, task_name in tasks:
        run_task(task_file, task_name)
        
    print("\n所有任务执行完成！")
    print("\n可以查看生成的可视化结果文件: user_ratings_distribution.png, svd_predictions_scatter.png, svd_factors_rmse.png") 