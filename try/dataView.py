import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("TkAgg")

# 读取数据
data = pd.read_csv("../data.csv", header=0).iloc[:, 1:]

# 设置一个最小显示的类别占比阈值
threshold = 0.01

# 获取目标值为 1、2、3 的子数据
target_values = [0, 1, 2]

# 创建一个图形，计算需要的子图数
cols = 3
rows = -(-len(data.keys()) // cols)

# 为每个 target 创建一个子图
for target in target_values:
    target_data = data[data['target'] == target].iloc[:, :-1]
    # 创建一个新的图形
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), constrained_layout=True)
    fig.suptitle(f'Distribution of Labels for target {target}', fontsize=16)

    # 遍历每列
    for idx, (label, ax) in enumerate(zip(target_data.keys(), axes.flat)):
        target_counts = target_data[label].value_counts(normalize=True)
        grouped_counts = target_counts.copy()

        # 将稀疏值归为“其他”
        grouped_counts["Other"] = grouped_counts[grouped_counts < threshold].sum()
        grouped_counts = grouped_counts[grouped_counts >= threshold]

        # 检查是否有数据
        if not grouped_counts.empty:
            # 绘制子图
            bars = grouped_counts.sort_values(ascending=False).plot(
                kind='bar', ax=ax, color='skyblue', alpha=0.8, edgecolor='black'
            )

            # 在每个条形图的顶部标注占比，便于对比
            for p in bars.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2%}',
                            (p.get_x() + p.get_width() / 2, height),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

            ax.set_title(f'{label} Value Proportions', fontsize=14)
            ax.tick_params(axis='x', labelrotation=45, labelsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1)
            # 设置y轴刻度单位为25%
            ax.set_yticks([i * 0.25 for i in range(5)])
            ax.set_yticklabels([f"{i * 25}%" for i in range(5)])
        else:
            # 如果没有数据，隐藏该子图
            ax.set_visible(False)
            ax.set_title(f'{label} (No Data)', fontsize=14)

    # 删除多余的空白子图
    for extra_ax in axes.flat[len(target_data.keys()):]:
        extra_ax.set_visible(False)

# 显示每个 target 的图形
plt.show()
