import torch
from torch_geometric.data import Dataset, HeteroData
import xarray as xr
import numpy as np

class CAMSDataset(Dataset):
    def __init__(self, file_path, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)

        self.ds = xr.open_dataset(file_path)
        self.lats = self.ds['latitude'].values
        self.lons = self.ds['longitude'].values
        self.times = self.ds['time'].values
        self.var = self.ds['XCO2'].values  # [T, lat, lon]

        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        self.pos = np.stack([lon_grid.flatten(), lat_grid.flatten()], axis=1)
        self.num_nodes = self.pos.shape[0]

    def len(self):
        return len(self.times)

    def get(self, idx):
        data = HeteroData()

        # grid 节点
        grid_feat = self.var[idx].reshape(-1, 1)  # [N,1] 这是 XCO2
        data['grid'].x = torch.tensor(grid_feat, dtype=torch.float32)
        data['grid'].pos = torch.tensor(self.pos, dtype=torch.float32)

        # time 节点 - 这里放当天的 XCO2 全局均值（用于训练 label / 校验）
        xco2_mean = float(grid_feat.mean())
        data['time'].x = torch.tensor([[xco2_mean]], dtype=torch.float32)
        # time.pos 放当天的中心经纬（可选）
        center_lon = float(self.pos[:, 0].mean())
        center_lat = float(self.pos[:, 1].mean())
        data['time'].pos = torch.tensor([[center_lon, center_lat]], dtype=torch.float32)

        # grid-grid 邻接（4-neighbor）
        lat_n, lon_n = len(self.lats), len(self.lons)
        edges = []
        for i in range(lat_n):
            for j in range(lon_n):
                u = i * lon_n + j
                if i + 1 < lat_n:
                    edges.append((u, (i + 1) * lon_n + j))
                if j + 1 < lon_n:
                    edges.append((u, i * lon_n + (j + 1)))
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['grid', 'spatial', 'grid'].edge_index = edge_index

        # grid -> time
        grid2time = torch.stack([
            torch.arange(self.num_nodes, dtype=torch.long),
            torch.zeros(self.num_nodes, dtype=torch.long)
        ], dim=0)
        data['grid', 'belong_to', 'time'].edge_index = grid2time

        # time temporal edge placeholder (窗口拼接时会构建真正的时序边)
        data['time', 'temporal', 'time'].edge_index = torch.zeros((2, 0), dtype=torch.long)

        return data
