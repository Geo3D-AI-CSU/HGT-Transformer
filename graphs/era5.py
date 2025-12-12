# era5_dataset.py
import os, glob
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch_geometric.data import Dataset, HeteroData


class ERA5Dataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(None)

        files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
        if not files:
            raise FileNotFoundError(f"No ERA5 files found in {data_dir}")

        # 合并 ERA5 文件
        self.ds = xr.open_mfdataset(files, combine="by_coords")

        # 选取变量：你可按需修改
        self.u10 = self.ds["u10"]       # (time, lat, lon)
        self.v10 = self.ds["v10"]       # (time, lat, lon)

        # 时间、空间维度
        self.times = pd.to_datetime(self.u10["valid_time"].values)
        self.lats = self.ds["latitude"].values
        self.lons = self.ds["longitude"].values

        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)
        self.num_nodes = self.n_lat * self.n_lon

        # 计算网格坐标
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        self.grid_pos = np.stack([lon_grid.flatten(), lat_grid.flatten()], axis=1)

        # 按天分组
        self.grouped = pd.Series(np.arange(len(self.times)), index=self.times).groupby(self.times.date)
        self.dates = list(self.grouped.groups.keys())

        # 预构建 4-neighbor 空间边
        self.grid_edges = self._build_spatial_edges(self.n_lat, self.n_lon)

    def _build_spatial_edges(self, n_lat, n_lon):
        edges = []
        def nid(i, j): return i * n_lon + j
        for i in range(n_lat):
            for j in range(n_lon):
                u = nid(i, j)
                if i + 1 < n_lat: edges.append((u, nid(i + 1, j)))
                if j + 1 < n_lon: edges.append((u, nid(i, j + 1)))
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        edges = np.array(edges, dtype=np.int64).T
        return torch.tensor(edges, dtype=torch.long)

    def len(self):
        return len(self.dates)

    def get(self, idx):
        """返回单日结构，用于 window 拼接"""
        date = self.dates[idx]
        time_idxs = self.grouped.get_group(date).values

        # 当天平均风场
        u_day = self.u10.isel(valid_time=time_idxs).mean(dim="valid_time").values.flatten().astype(np.float32)
        v_day = self.v10.isel(valid_time=time_idxs).mean(dim="valid_time").values.flatten().astype(np.float32)

        doy = np.float32(pd.to_datetime(date).dayofyear)

        # === grid 节点 ===
        grid_feat = np.stack([u_day, v_day, np.full_like(u_day, doy)], axis=1)
        data = HeteroData()
        data["grid"].x = torch.tensor(grid_feat, dtype=torch.float32)
        data["grid"].pos = torch.tensor(self.grid_pos, dtype=torch.float32)

        # === time 节点 ===
        data["time"].x = torch.tensor([[doy]], dtype=torch.float32)
        data["time"].pos = torch.tensor([[0.0, float(idx)]], dtype=torch.float32)

        # === grid-grid edges ===
        data["grid", "spatial", "grid"].edge_index = self.grid_edges

        # === grid -> time belong_to ===
        g2t = torch.stack([
            torch.arange(self.num_nodes, dtype=torch.long),
            torch.zeros(self.num_nodes, dtype=torch.long)
        ], dim=0)
        data["grid", "belong_to", "time"].edge_index = g2t

        # 不构建 time-time 边（由窗口自动构建）
        data["time", "temporal", "time"].edge_index = torch.zeros((2, 0), dtype=torch.long)

        return data
