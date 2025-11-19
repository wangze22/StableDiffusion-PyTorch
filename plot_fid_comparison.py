import matplotlib.pyplot as plt
import numpy as np

# 3个模型的名称
models = ['DiT_9L', 'DiT_12L', 'Unet']

# FID 数据
fid_ideal = [12.6431, 11.7593, 14.7716]  # 理想条件下的FID
fid_noise = [12.4391, 11.9823, 16.0348]  # 噪声条件下的FID

# 设置柱状图位置
x = np.arange(len(models))
width = 0.35

# 创建图表
fig, ax = plt.subplots(figsize = (10, 6))

# 绘制柱状图
bars1 = ax.bar(x - width / 2, fid_ideal, width, label = 'Ideal', color = '#2E86AB', alpha = 0.8)
bars2 = ax.bar(x + width / 2, fid_noise, width, label = 'Noise', color = '#A23B72', alpha = 0.8)

# 在柱状图上添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{height:.2f}',
            ha = 'center', va = 'bottom', fontsize = 10,
            )

# 设置图表属性
ax.set_xlabel('Models', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('FID Score', fontsize = 12, fontweight = 'bold')
ax.set_title('FID Comparison: Ideal vs Noise Conditions', fontsize = 14, fontweight = 'bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize = 11)
ax.grid(axis = 'y', alpha = 0.3, linestyle = '--')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('fid_comparison.png', dpi = 300, bbox_inches = 'tight')
print("图表已保存为 fid_comparison.png")

# 显示图表
plt.show()
