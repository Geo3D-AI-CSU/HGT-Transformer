import os
import re
import datetime
import glob
import numpy as np
import torch
from torch_geometric.data import HeteroData, Dataset
import xarray as xr
from scipy.spatial import cKDTree

# 默认半径（度）
RADIUS_DEG = 0.1
# 每个观测点最多保留的邻居数（防止邻居数骤增导致 OOM / MemoryError）
MAX_NEIGHBORS = 50


def list_all_files(data_dir):
    pattern = os.path.join(data_dir, "oco2_LtCO2_*.nc*")
    return sorted(glob.glob(pattern))


def parse_date_from_filename(fp):
    base = os.path.basename(fp)
    # 尝试提取连续 6 位数字（yyMMdd）
    m = re.search(r'oco2_LtCO2_(\d{6})_', base)
    if not m:
        return None
    yymmdd = m.group(1)
    try:
        dt = datetime.datetime.strptime(yymmdd, "%y%m%d")
        # 将年份补成 20xx（因为 200101 表示 2020-01-01）
        # 注意：如果文件来自 2100 年以后的数据，这里需要修改
        if dt.year < 1970:  # 极端兜底
            dt = dt.replace(year=dt.year + 2000)
        return dt.date()
    except Exception:
        return None


def scan_and_report_files(data_dir):

    files = list_all_files(data_dir)
    file_map = {}
    dates = []
    for f in files:
        d = parse_date_from_filename(f)
        if d is not None:
            # 若一天有多个文件，选择第一个（或可根据策略合并，这里选择第一个）
            if d not in file_map:
                file_map[d] = f
                dates.append(d)

    if not dates:
        print(f"❌ 未找到任何符合命名约定的 OCO-2 文件：{data_dir}")
        return [], {}

    dates_sorted = sorted(dates)
    start = dates_sorted[0]
    end = dates_sorted[-1]
    full = []
    cur = start
    while cur <= end:
        full.append(cur)
        cur = cur + datetime.timedelta(days=1)

    missing = [d for d in full if d not in file_map]

    print(f"📆 共检测到 {len(dates_sorted)} 天数据，从 {start} 到 {end}")
    if missing:
        print(f"⚠️ 缺失 {len(missing)} 天数据，示例: {[d.isoformat() for d in missing[:5]]} ...")
        try:
            with open(os.path.join(data_dir, "missing_dates.txt"), "w", encoding="utf-8") as fh:
                for d in missing:
                    fh.write(d.isoformat() + "\n")
        except Exception as e:
            print("写入 missing_dates.txt 失败：", e)
    else:
        print("✅ 没有缺失日期！")

    return dates_sorted, file_map


def build_obs_edges_kdtree(lat, lon, radius_deg=RADIUS_DEG, max_neighbors=MAX_NEIGHBORS):

    if len(lat) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    coords = np.vstack([lat, lon]).T
    tree = cKDTree(coords)

    # query_ball_point 返回每个点的邻居索引列表
    neighbors = tree.query_ball_point(coords, r=radius_deg, return_sorted=True)

    rows = []
    cols = []
    for i, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        # 排除自身
        nbrs = [j for j in nbrs if j != i]
        if len(nbrs) == 0:
            continue
        # 限制邻居数，选择最近的 max_neighbors（基于距离）
        if max_neighbors is not None and len(nbrs) > max_neighbors:
            # 计算距离并选最近的若干
            dists = np.sqrt((coords[nbrs, 0] - coords[i, 0]) ** 2 + (coords[nbrs, 1] - coords[i, 1]) ** 2)
            idx_sorted = np.argsort(dists)[:max_neighbors]
            nbrs = [nbrs[k] for k in idx_sorted]

        for j in nbrs:
            rows.append(i)
            cols.append(j)

    if len(rows) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = np.vstack([rows, cols]).astype(np.int64)
    # 为确保对称性，也加入反向边（如果未存在）
    # 先构建 set 以便快速查重
    edges_set = set((int(a), int(b)) for a, b in zip(edge_index[0], edge_index[1]))
    extra_rows = []
    extra_cols = []
    for a, b in zip(edge_index[0], edge_index[1]):
        if (int(b), int(a)) not in edges_set:
            extra_rows.append(int(b))
            extra_cols.append(int(a))
            edges_set.add((int(b), int(a)))
    if extra_rows:
        edge_index = np.hstack([edge_index, np.vstack([extra_rows, extra_cols])])

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    return edge_index_tensor


def safe_read_variable(ds, name):
    """从 xarray.Dataset 或 netCDF4 Dataset 安全读取变量为 numpy array"""
    try:
        arr = ds[name][:]
        return np.array(arr)
    except Exception:
        # 返回空数组，长度 0
        return np.array([])


def build_hetero_from_file(fp, radius_deg=RADIUS_DEG, max_neighbors=MAX_NEIGHBORS):

    date = parse_date_from_filename(fp)
    if date is None:
        print(f"⚠️ 无法从文件名解析日期，跳过：{fp}")
        return None

    try:
        ds = xr.open_dataset(fp)
    except Exception as e:
        print(f"❌ 无法打开 NetCDF 文件 {fp}: {e}")
        return None

    try:
        lat = safe_read_variable(ds, 'latitude').flatten()
        lon = safe_read_variable(ds, 'longitude').flatten()
        xco2 = safe_read_variable(ds, 'xco2').flatten()
        # quality flag 如果存在，可做筛选（这里不强制）
        qf = safe_read_variable(ds, 'xco2_quality_flag').flatten()
    except Exception as e:
        print(f"❌ 读取变量失败 {fp}: {e}")
        ds.close()
        return None

    # 只保留有效点：非 NaN，且（如果存在 quality flag）qf == 0
    mask = np.ones_like(lat, dtype=bool)
    if lat.size != lon.size or lat.size != xco2.size:
        # 形状不一致时，裁剪到最小长度（并警告）
        n = min(lat.size, lon.size, xco2.size)
        lat = lat[:n]
        lon = lon[:n]
        xco2 = xco2[:n]
        mask = np.ones(n, dtype=bool)
        print(f"⚠️ 数据长度不一致，已裁剪到长度 {n}：{fp}")

    if qf.size == lat.size:
        mask = mask & (qf == 0)

    mask = mask & (~np.isnan(lat)) & (~np.isnan(lon)) & (~np.isnan(xco2))

    if mask.sum() == 0:
        # 没有有效观测：返回一个空的 HeteroData（符合上层期待）
        data = HeteroData()
        data['obs'].x = torch.zeros((0, 4), dtype=torch.float32)
        data['obs'].pos = torch.zeros((0, 2), dtype=torch.float32)
        data['time'].x = torch.zeros((1, 4), dtype=torch.float32)
        data['time'].pos = torch.zeros((1, 2), dtype=torch.float32)
        data['time'].date = np.datetime64(date)
        ds.close()
        print(f"[{date}] 无有效观测点，返回空图。")
        return data

    lat = lat[mask].astype(np.float32)
    lon = lon[mask].astype(np.float32)
    xco2 = xco2[mask].astype(np.float32)

    N = lat.shape[0]
    doy = np.array([date.timetuple().tm_yday], dtype=np.float32)[0]

    print(f"📅 正在处理日期：{date.isoformat()}，文件名：{os.path.basename(fp)}")
    print(f"共有 {N} 个观测点")
    print(f"🔧 使用 KDTree 构建空间边 (radius={radius_deg}°，max_neighbors={max_neighbors})...")

    try:
        obs_edge_index = build_obs_edges_kdtree(lat, lon, radius_deg=radius_deg, max_neighbors=max_neighbors)
        print(f"[完成] 共构建 {obs_edge_index.shape[1]} 条空间边。")
    except Exception as e:
        print(f"❌ 构建空间边失败: {e}")
        ds.close()
        return None

    # 构建 HeteroData
    data = HeteroData()
    # obs 节点特征： lat, lon, xco2, doy
    obs_x = np.stack([lat, lon, xco2, np.full(N, doy, dtype=np.float32)], axis=1)
    data['obs'].x = torch.tensor(obs_x, dtype=torch.float32)
    data['obs'].pos = torch.tensor(np.stack([lat, lon], axis=1), dtype=torch.float32)
    data['obs', 'spatial', 'obs'].edge_index = obs_edge_index

    # time 节点（聚合）
    lat_mean = float(lat.mean())
    lon_mean = float(lon.mean())
    xco2_mean = float(xco2.mean())
    time_x = np.array([[lat_mean, lon_mean, xco2_mean, doy]], dtype=np.float32)
    data['time'].x = torch.tensor(time_x, dtype=torch.float32)
    data['time'].pos = torch.tensor([[lat_mean, lon_mean]], dtype=torch.float32)
    data['time'].date = np.datetime64(date)

    # obs -> time 边，所有 obs 指向 time 节点 0
    if N > 0:
        obs2time = np.vstack([np.arange(N, dtype=np.int64), np.zeros(N, dtype=np.int64)])
        data['obs', 'belong_to', 'time'].edge_index = torch.tensor(obs2time, dtype=torch.long)
    else:
        data['obs', 'belong_to', 'time'].edge_index = torch.zeros((2, 0), dtype=torch.long)

    ds.close()
    print(f"[完成] 日期 {date.isoformat()} 的图数据构建完毕 ✅")
    return data


class OCO2Dataset(Dataset):

    def __init__(self, root, radius_deg=RADIUS_DEG, max_neighbors=MAX_NEIGHBORS, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root_dir = root
        self.radius_deg = radius_deg
        self.max_neighbors = max_neighbors

        self.available_dates, self.date_to_file = self._scan_files()
        # ✅ 检查日期顺序并打印确认
        self.available_dates = sorted(self.available_dates)
        print("✅ OCO2Dataset 已按时间顺序加载。前5天示例：")
        print([d.isoformat() for d in self.available_dates[:5]])
        print(f"共 {len(self.available_dates)} 天，最后一天是 {self.available_dates[-1].isoformat()}")

    def _scan_files(self):
        dates_sorted, file_map = scan_and_report_files(self.root_dir)
        file_map_by_date = {}
        for d, fp in file_map.items():
            file_map_by_date[d] = fp
        return dates_sorted, file_map_by_date

    def len(self):
        return len(self.available_dates)

    def __len__(self):
        return self.len()

    def get(self, idx):

        if idx < 0 or idx >= len(self.available_dates):
            raise IndexError("索引越界")

        date = self.available_dates[idx]
        fp = self.date_to_file.get(date, None)
        if fp is None or not os.path.exists(fp):
            print(f"⚠️ 缺失或找不到文件：{date.isoformat()}，返回空图并跳过。")
            # 返回空 HeteroData
            empty = HeteroData()
            empty['obs'].x = torch.zeros((0, 4), dtype=torch.float32)
            empty['obs'].pos = torch.zeros((0, 2), dtype=torch.float32)
            empty['time'].x = torch.zeros((1, 4), dtype=torch.float32)
            empty['time'].pos = torch.zeros((1, 2), dtype=torch.float32)
            empty['time'].date = np.datetime64(date)
            return empty

        data = build_hetero_from_file(fp, radius_deg=self.radius_deg, max_neighbors=self.max_neighbors)
        if data is None:
            # 出错时返回空 HeteroData（避免上层报错）
            empty = HeteroData()
            empty['obs'].x = torch.zeros((0, 4), dtype=torch.float32)
            empty['obs'].pos = torch.zeros((0, 2), dtype=torch.float32)
            empty['time'].x = torch.zeros((1, 4), dtype=torch.float32)
            empty['time'].pos = torch.zeros((1, 2), dtype=torch.float32)
            empty['time'].date = np.datetime64(date)
            return empty

        return data


if __name__ == "__main__":
    # 简单测试用例（替换为你的路径）
    DATA_DIR = r"D:/GCN-Transformer/data/OCO-2"
    ds = OCO2Dataset(DATA_DIR, radius_deg=0.1, max_neighbors=50)
    print(f"OCO-2 共 {len(ds)} 天")
    if len(ds) > 0:
        sample = ds[0]
        print(sample)
        print("Obs nodes:", sample['obs'].x.shape[0])
        print("Time nodes:", sample['time'].x.shape[0])
