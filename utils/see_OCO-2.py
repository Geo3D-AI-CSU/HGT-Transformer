import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

from matplotlib.colors import LinearSegmentedColormap

# =====================
# 文件路径设置
# =====================
oco2_file = "D:/GCN-Transformer/data/OCO-2/oco2_LtCO2_200101_B11100Ar_230603192102s.nc4"
output_dir = "D:/GCN-Transformer/results/"
os.makedirs(output_dir, exist_ok=True)

colors = [
    (1.0, 1.0, 1.0),   # 白色
    (1.0, 0.6, 0.6),   # 浅红
    (0.8, 0.0, 0.0)    # 深红
]
cmap_red = LinearSegmentedColormap.from_list("white_to_red", colors)
vmin = 400
vmax = 420
# =====================
# 1️⃣ 仅加载必要变量 + 按需读取
# =====================
ds = xr.open_dataset(
    oco2_file,
    chunks={'sounding_id': 10000},  # 分块读取避免爆内存
    decode_times=True
)[['xco2', 'longitude', 'latitude', 'time']]

print(ds)

# =====================
# 2️⃣ 转换为 DataFrame（只取有效点）
# =====================
oco2 = ds[['longitude', 'latitude', 'xco2', 'time']].to_dask_dataframe().compute()
oco2 = oco2.dropna(subset=['xco2', 'longitude', 'latitude'])

# =====================
# 3️⃣ 筛选特定日期（例如 2020-07-15）
# =====================
oco2['time'] = pd.to_datetime(oco2['time'])
oco2_day = oco2[oco2['time'].dt.date == pd.Timestamp('2020-01-01').date()]

print(f"✅ Selected {len(oco2_day)} valid OCO-2 soundings for 2020-01-01")

# =====================
# 4️⃣ 绘制全球分布散点图
# =====================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(10, 4))
ax = plt.axes(projection=ccrs.Robinson())

sc = plt.scatter(
    oco2_day['longitude'], oco2_day['latitude'],
    c=oco2_day['xco2'], s=3,
    cmap=cmap_red,
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree()
)

ax.coastlines(linewidth=0.5)
ax.set_global()

cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05, fraction=0.05)
cbar.set_label('XCO₂ (ppm)')

ax.set_title('Global OCO-2 XCO₂ observations\n(January 1st,2020)', fontsize=10, pad=8)
plt.tight_layout()

# =====================
# 5️⃣ 保存高分辨率图像
# =====================
plt.savefig(f"{output_dir}OCO2.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figure saved successfully (no memory issue).")
