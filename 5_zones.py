import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件（请根据实际路径修改）
df = pd.read_csv('/mnt/private1/luansl/0Results_for_paper/CH4_from_water/Data/Average_flux_rate/Overall_Emission_Stats.csv')

# 定义解析函数，将 "mean±sd" 格式字符串拆分为均值和标准差
def parse_value(s):
    try:
        mean_str, sd_str = s.split('±')
        return float(mean_str), float(sd_str)
    except Exception as e:
        return np.nan, np.nan

# 对各列解析出均值和标准差
df[['Lake_mean', 'Lake_sd']] = df['Lake'].apply(lambda s: pd.Series(parse_value(s)))
df[['Reservoir_mean', 'Reservoir_sd']] = df['Reservoir'].apply(lambda s: pd.Series(parse_value(s)))
df[['Combined_mean', 'Combined_sd']] = df['Combined'].apply(lambda s: pd.Series(parse_value(s)))

# 绘制分组柱状图
x = np.arange(len(df))  # 每个区域对应一个位置
width = 0.25  # 每组柱子的宽度

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制三组柱状图并添加误差棒
bars_lake = ax.bar(x - width, df['Lake_mean'], width, yerr=df['Lake_sd'], capsize=5, label='Lake')
bars_res = ax.bar(x, df['Reservoir_mean'], width, yerr=df['Reservoir_sd'], capsize=5, label='Reservoir')
bars_comb = ax.bar(x + width, df['Combined_mean'], width, yerr=df['Combined_sd'], capsize=5, label='Combined')

# 设置 x 轴刻度与标签
ax.set_xticks(x)
ax.set_xticklabels(df['Zone_5lakes'])
# 设置坐标轴刻度数字大小
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylabel('CH$_4$ Flux((mg C/m$^2$/d)', fontsize=18)

ax.legend(fontsize=18)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

# 添加左上角标签
caption = '(b)'
ax.text(0.05, 0.9, caption, fontsize=20, transform=ax.transAxes, va='center', ha='center')

plt.tight_layout()
plt.show()