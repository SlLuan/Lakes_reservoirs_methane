import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.plot import show
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.mpl.geoaxes as geoaxes
import os
from matplotlib.pyplot import FuncFormatter

def get_levels(data_masked, method='auto', num_levels=11, manual_levels=None):

    if method == 'auto':
        vmin = np.nanmin(data_masked)
        vmax = np.nanmax(data_masked)
        levels = np.linspace(vmin, vmax, num_levels)
        levels = np.round(levels).astype(int)
    elif method == 'percentile':
        levels = np.percentile(data_masked.compressed(), np.linspace(0, 100, num_levels))
    elif method == 'manual':
        if manual_levels is None:
            raise ValueError(" manual_levels 列表，当 method 设置为 'manual' 。")
        levels = np.array(manual_levels)
    else:
        raise ValueError( "method 参数，请选择 'auto', 'percentile' 或 'manual'.")
    return levels


def plot_LCT_flux_map(data_masked, transform, levels, legend_label, caption, ax):
    china_main = gpd.read_file('/mnt/private1/luansl/China_CH4_from_Inland_water/Data/Shengji.shp')
    china_nine = gpd.read_file('/mnt/private1/luansl/China_CH4_from_Inland_water/Data/nanhait.shp')

    if china_nine.crs is None:
        print("china_nine 没有定义坐标系，正在设置...")
        china_nine = china_nine.set_crs('epsg:4326')  

    china_main = china_main.to_crs('epsg:4326')
    china_nine = china_nine.to_crs('epsg:4326')

    # 加载湖区数据，并设置坐标系
    lakezones = gpd.read_file('/mnt/private1/luansl/0Results_for_paper/CH4_from_water/Data/SHP/Lake_Zones_5.shp')
    if lakezones.crs is None:
        lakezones = lakezones.set_crs('epsg:4326')
    else:
        lakezones = lakezones.to_crs('epsg:4326')

    rows, cols = data_masked.shape
    # 根据仿射变换生成网格坐标
    x = np.linspace(transform[2], transform[2] + transform[0] * cols, cols)
    y = np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
    lon, lat = np.meshgrid(x, y)

    # 设置投影，这里使用 LambertConformal
    proj = ccrs.LambertConformal(central_longitude=105, standard_parallels=(25, 47))

    # 设置显示范围
    ax.set_extent([75, 135, 18, 54], crs=ccrs.PlateCarree())
    ax.set_facecolor('white')

    # 添加子图编号（在左上角显示 caption）
    ax.text(0.05, 0.9, caption, fontsize=20, transform=ax.transAxes, va='center', ha='center')

    norm = BoundaryNorm(levels, ncolors=256)
    cf = ax.contourf(lon, lat, data_masked, levels=levels, extend='both', cmap='RdYlBu_r',
                     norm=norm, transform=ccrs.PlateCarree())

    # 绘制中国行政边界
    china_main.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.3, zorder=2,
                    transform=ccrs.PlateCarree())
    # 绘制南海区域边界
    china_nine.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.3, zorder=3,
                    transform=ccrs.PlateCarree())
    # 绘制湖区轮廓（仅显示边界，并加粗线宽）
    lakezones.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, zorder=4,
                   transform=ccrs.PlateCarree())

    # 添加网格线及格式化
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        linestyle='--',
        alpha=0.5,
        x_inline=False,
        y_inline=False,
        xpadding=10,
        ypadding=10,
        rotate_labels=False,
    )
    gl.top_labels = False
    # gl.left_labels = False
    # gl.bottom_labels = False
    gl.right_labels = False
    
    # 设置网格线标签的字体大小
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    gl.xformatter = plt.FuncFormatter(lambda x, pos: f'{int(x)}°E')
    gl.yformatter = plt.FuncFormatter(lambda y, pos: f'{int(y)}°N')

    # 在右下角创建南海插图
    ax_inset = inset_axes(
        ax,
        width="20%",
        height="20%",
        loc='lower right',
        bbox_to_anchor=(0.0, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
        axes_class=geoaxes.GeoAxes,
        axes_kwargs={'projection': ccrs.LambertConformal(central_longitude=105)}
    )
    ax_inset.set_extent([105, 120, 0, 23], crs=ccrs.PlateCarree())
    ax_inset.contourf(lon, lat, data_masked, levels=levels, cmap='RdYlBu_r',
                      norm=norm, transform=ccrs.PlateCarree())
    china_main.plot(ax=ax_inset, edgecolor='black', facecolor='none', linewidth=0.3, zorder=2,
                    transform=ccrs.PlateCarree())
    china_nine.plot(ax=ax_inset, edgecolor='black', facecolor='none', linewidth=0.3, zorder=3,
                    transform=ccrs.PlateCarree())
    # 插图中也绘制湖区轮廓
    lakezones.plot(ax=ax_inset, edgecolor='black', facecolor='none', linewidth=0.6, zorder=4,
                   transform=ccrs.PlateCarree())

    # 在子图内部添加颜色条
    cax = inset_axes(
        ax,
        width="40%",
        height="3%",
        loc='lower left',
        bbox_to_anchor=(0.01, 0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    # 使用 set_xlabel 设置颜色条标签，并指定字体大小
    cbar.ax.set_xlabel(legend_label, fontsize=16, labelpad=35)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=legend_label, labelpad=35)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}' if x % 1 == 0 else f'{x}'))
    cbar.ax.xaxis.set_label_coords(0.5, 2.3)
    
# 主程序部分：读取 TIFF 文件并调用绘图函数
# tif_file = '/mnt/private1/luansl/0Results_for_paper/CH4_from_water/Data/LAK/CNGHGv1.0_CH4_AFOLU_LAK_Flux_mean_2000_2023.tif'#b
# tif_file = '/mnt/private1/luansl/0Results_for_paper/CH4_from_water/Data/LAK_RSV/CNGHGv1.0_CH4_AFOLU_LAK_RSV_Flux_mean_2000_2023.tif'#a
tif_file = '/mnt/private1/luansl/0Results_for_paper/CH4_from_water/Data/RSV/CNGHGv1.0_CH4_AFOLU_RSV_Flux_mean_2000_2023.tif'#c
with rasterio.open(tif_file) as src:
    data = src.read(1).astype(np.float32)
    transform = src.transform
    nodata = src.nodata

# 对数据进行掩膜处理（假设 nodata 为 -9999）
data_masked = np.ma.masked_equal(data, nodata)
data_masked = np.ma.masked_equal(data_masked, 0)

# 三种方式生成 levels：
# 1. 自动化均分（auto）
levels_auto = get_levels(data_masked, method='auto', num_levels=11)
# 2. 百分比划分（percentile）
levels_percentile = get_levels(data_masked, method='percentile', num_levels=11)
# 3. 全手动（manual），例如手动设置级数
# manual_levels = [np.nanmin(data_masked),50, 75, 150, 200, 250, 325, 400, 500, 800, 1600, 3200, 5000, 5500, 6500]
# manual_levels = [0.5, 10, 15, 40, 60, 70, 85, 95, 105, 120, 130, 140, 145]  #b
# manual_levels = [20, 25, 30, 40, 45, 60, 75, 95, 110, 120, 140, 150, 165] #a 
manual_levels = [10, 20, 30, 35, 40, 45, 60, 70, 90, 110, 130, 150, 170] #c
levels_manual = get_levels(data_masked, method='manual', manual_levels=manual_levels)

# 选择一种方式作为色带划分（可切换为 levels_auto 或 levels_percentile 或 levels_manual）
levels = levels_manual

legend_label = 'CH$_4$ Flux ($\mathrm{mg}$ C/$\mathrm{m}^2$/day)'
caption = '(a)'

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertConformal(central_longitude=105, standard_parallels=(25,47)))

plot_LCT_flux_map(data_masked, transform, levels, legend_label, caption, ax)

plt.tight_layout()

plt.show()
