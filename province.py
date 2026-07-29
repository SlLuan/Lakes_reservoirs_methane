import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, FixedLocator

# 自定义坐标轴变换函数（核心逻辑）
def transform(x):
    """ 将原始数据映射到等间距坐标 """
    x = np.asarray(x)
    return np.where(x <= 0.2, 
                    x * 0.7/0.5,           # 0-0.5 映射到 0-0.7（占70%宽度）
                    0.7 + (x-0.5)*0.3/1.5  # 0.5-2 映射到 0.7-1.0（占30%宽度）
                   )

def inverse_transform(x):
    """ 逆变换：从等间距坐标映射回原始数据 """
    x = np.asarray(x)
    return np.where(x <= 0.7, 
                    x * 0.5/0.7,           # 0-0.7 映射回 0-0.5
                    0.5 + (x-0.7)*1.5/0.3  # 0.7-1.0 映射回 0.5-2
                   )

# 加载数据
file_path = '/mnt/private1/luansl/China_CH4_from_Inland_water/Data/EXCEL/00-23_Province_mean.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 创建画布和坐标轴，应用自定义变换
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xscale('function', functions=(transform, inverse_transform))  # 关键步骤

# 定义y轴位置
y = np.arange(len(data['Province']))
width = 0.6

# 绘制堆叠柱状图（需先变换数据）
rsv_trans = transform(data['RSV']/1e6)
lak_trans = transform(data['LAK']/1e6)
ax.barh(y, rsv_trans, width, color='skyblue', label='Reservoir')
ax.barh(y, lak_trans, width, left=rsv_trans, color='lightgreen', label='Lake')

# 设置 y 轴刻度标签为 Province 名称，并设置字体大小
ax.set_yticks(y)
ax.set_yticklabels(data['Province'], fontsize=14)

# 设置 x 轴刻度位置（转换后的值）及标签（显示原始数值）
ticks_orig = np.array([0, 0.1, 0.2, 2])
ticks_trans = transform(ticks_orig)
ax.xaxis.set_major_locator(FixedLocator(ticks_trans))

def custom_formatter(x, pos):
    original_value = inverse_transform(x)
    # 若值接近整数，则显示整数（不带小数点）
    if np.isclose(original_value, np.round(original_value), atol=1e-2):
        return f"{int(np.round(original_value))}"
    else:
        return f"{original_value:.1f}"

ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax.tick_params(axis='x', labelsize=16)  # 自定义 x 轴数字的字体大小

# 坐标轴标签和图例
ax.set_xlabel('CH$_4$ Emissions (Tg CH$_4$/ yr)', fontsize=16)
ax.legend(loc='lower left', frameon=False, fontsize=16, markerscale=1.5)
caption = '(d)'
ax.text(0.05, 0.9, caption, fontsize=20, transform=ax.transAxes, va='center', ha='center')

# 将纵坐标（y轴）放在右侧
ax.yaxis.tick_right()
ax.yaxis.set_label_position("left")
ax.invert_xaxis()

plt.tight_layout()
plt.savefig('/mnt/private1/luansl/0Results_for_paper/CH4_from_water/Figures/Figure4/Figure(d).png', dpi=300, bbox_inches='tight')
plt.show()